import argparse
import asyncio
import multiprocessing
import os
import re
import sys
from contextlib import asynccontextmanager
from io import TextIOWrapper
from pathlib import Path

import uvicorn
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    ModelLoadError,
    StyleNotFoundError,
    SynthesisError,
)
from fastapi import (
    FastAPI,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from packaging.version import Version

from voicevox_engine import __version__
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.coeiroink_v2.catalog import OfficialSiteCatalogClient
from voicevox_engine.coeiroink_v2.dictionary import (
    set_dictionary as set_coeiroink_dictionary,
)
from voicevox_engine.coeiroink_v2.metadata import (
    SpeakerMetadataStore,
)
from voicevox_engine.coeiroink_v2.router import create_v2_router
from voicevox_engine.engine_manifest import EngineManifestLoader
from voicevox_engine.metas.metas_store import MetasStore
from voicevox_engine.preset import PresetManager
from voicevox_engine.setting import (
    USER_SETTING_PATH,
    CorsPolicyMode,
    SettingLoader,
)
from voicevox_engine.synthesis_engine import SynthesisEngineBase, make_synthesis_engines
from voicevox_engine.synthesis_engine.make_synthesis_engines import resolve_device
from voicevox_engine.user_dict import (
    update_dict,
)
from voicevox_engine.utility import (
    engine_root,
)
from voicevox_engine.voicevox_compat.mutable_api import (
    boolean_from_env as decide_boolean_from_env,
)
from voicevox_engine.voicevox_compat.mutable_api import mutability_guard
from voicevox_engine.voicevox_compat.resources import (
    ResourceManager,
)
from voicevox_engine.voicevox_compat.router import (
    VoicevoxRouterDependencies,
    create_voicevox_router,
)


def _version_key(version: str) -> Version:
    """Return a comparable Core version, including local suffixes such as +cpu."""
    return Version(version)


def _non_negative_int(value: str) -> int:
    """デバイス番号として利用できる0以上の整数を解析する。"""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("0以上の整数を指定してください。")
    return parsed


def set_output_log_utf8() -> None:
    """
    stdout/stderrのエンコーディングをUTF-8に切り替える関数
    """
    # コンソールがない環境だとNone https://docs.python.org/ja/3/library/sys.html#sys.__stdin__
    if sys.stdout is not None:
        # 必ずしもreconfigure()が実装されているとは限らない
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            # バッファを全て出力する
            sys.stdout.flush()
            sys.stdout = TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="backslashreplace"
            )
    if sys.stderr is not None:
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            sys.stderr.flush()
            sys.stderr = TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="backslashreplace"
            )


def _create_lifespan(cancellable_engine: CancellableEngine | None):
    """起動時の辞書更新と、任意のキャンセル監視タスクの寿命を管理する。"""

    cancellable_disconnection_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal cancellable_disconnection_task
        if cancellable_engine is not None:
            cancellable_disconnection_task = asyncio.create_task(
                cancellable_engine.catch_disconnection()
            )
        try:
            update_dict()
            yield
        finally:
            if cancellable_disconnection_task is not None:
                cancellable_disconnection_task.cancel()
                try:
                    await cancellable_disconnection_task
                except asyncio.CancelledError:
                    # lifespan終了時に自分で取り消した監視タスクの完了通知なので、異常終了として扱わない。
                    pass
                cancellable_disconnection_task = None

    return lifespan


def _add_exception_handlers(app: FastAPI) -> None:
    """Coreの公開例外をHTTPステータスへ変換する。"""

    @app.exception_handler(StyleNotFoundError)
    async def style_not_found_handler(
        _request: Request, err: StyleNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(err)})

    @app.exception_handler(ModelLoadError)
    @app.exception_handler(SynthesisError)
    async def synthesis_error_handler(
        _request: Request, err: RuntimeError
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(err)})

    @app.exception_handler(InvalidSynthesisParameterError)
    async def invalid_synthesis_parameter_handler(
        _request: Request, err: InvalidSynthesisParameterError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(err)})


def _add_cors_middleware(
    app: FastAPI,
    cors_policy_mode: CorsPolicyMode,
    allow_origin: list[str] | None,
) -> None:
    """CLI設定からCORS許可リストを作り、不許可Originを一箇所で遮断する。"""

    localhost_regex = "^https?://(localhost|127\\.0\\.0\\.1)(:[0-9]+)?$"
    compiled_localhost_regex = re.compile(localhost_regex)
    allowed_origins = ["*"]
    if cors_policy_mode == "localapps":
        allowed_origins = ["app://."]
        if allow_origin is not None:
            allowed_origins += allow_origin
            if "*" in allow_origin:
                print(
                    'WARNING: Deprecated use of argument "*" in allow_origin. '
                    'Use option "--cors_policy_mode all" instead. See "--help" for more.',
                    file=sys.stderr,
                )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_origin_regex=localhost_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def block_origin_middleware(request: Request, call_next):
        origin = request.headers.get("Origin")
        if (
            origin is None
            or "*" in allowed_origins
            or origin in allowed_origins
            or compiled_localhost_regex.fullmatch(origin)
        ):
            return await call_next(request)
        return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})


def generate_app(
    synthesis_engines: dict[str, SynthesisEngineBase],
    latest_core_version: str,
    setting_loader: SettingLoader,
    root_dir: Path | None = None,
    cors_policy_mode: CorsPolicyMode = CorsPolicyMode.localapps,
    allow_origin: list[str] | None = None,
    speaker_info_dir: Path | None = None,
    cancellable_engine: CancellableEngine | None = None,
    device: str | None = None,
    disable_mutable_api: bool = False,
) -> FastAPI:
    """Coreアダプター群をCOEIROINK v2 APIと`/voicevox`互換APIへ束ねたFastAPIアプリを構築する。"""

    if root_dir is None:
        root_dir = engine_root()
    if speaker_info_dir is None:
        speaker_info_dir = root_dir / "speaker_info"
    speaker_info_dir = speaker_info_dir.expanduser().resolve()
    if device is None:
        device = getattr(synthesis_engines[latest_core_version], "device", None)
    if device is None:
        raise ValueError(
            "音声合成エンジンが選択デバイスを公開していないため起動できません。"
        )
    device = resolve_device(device=device)

    app = FastAPI(
        title="COEIROINK Server OSS Edition",
        description=(
            "COEIROINK /v1 APIと、/voicevox配下のVOICEVOX互換会話音声APIを提供します。"
        ),
        version=__version__,
        lifespan=_create_lifespan(cancellable_engine),
    )
    _add_exception_handlers(app)
    _add_cors_middleware(app, cors_policy_mode, allow_origin)

    preset_manager = PresetManager(
        preset_path=root_dir / "presets.yaml",
    )
    engine_manifest_loader = EngineManifestLoader(
        root_dir / "engine_manifest.json", root_dir
    )

    metas_store = MetasStore(speaker_info_dir)
    speaker_metadata_store = SpeakerMetadataStore(
        speaker_info_dir, metas_store=metas_store
    )
    resource_manager = ResourceManager(speaker_info_dir)

    # `/v1`と`/voicevox`は同じ公開Core AudioManagerを共有し、互換層に別の推論器を持たせない。
    v2_audio_manager = getattr(
        synthesis_engines[latest_core_version], "audio_manager", None
    )
    if v2_audio_manager is not None:
        app.include_router(
            create_v2_router(
                audio_manager=v2_audio_manager,
                metadata_store=speaker_metadata_store,
                catalog=OfficialSiteCatalogClient(),
                dictionary_callback=set_coeiroink_dictionary,
                engine_version=__version__,
                device=device,
                default_processing_algorithm="td-psola",
            )
        )

    app.include_router(
        create_voicevox_router(
            VoicevoxRouterDependencies(
                synthesis_engines=synthesis_engines,
                latest_core_version=latest_core_version,
                preset_manager=preset_manager,
                engine_manifest_loader=engine_manifest_loader,
                metas_store=metas_store,
                speaker_metadata_store=speaker_metadata_store,
                resource_manager=resource_manager,
                setting_loader=setting_loader,
                cancellable_engine=cancellable_engine,
                verify_mutability_allowed=mutability_guard(disable_mutable_api),
            )
        )
    )
    return app


if __name__ == "__main__":
    multiprocessing.freeze_support()

    output_log_utf8 = os.getenv("VV_OUTPUT_LOG_UTF8", default="")
    if output_log_utf8 == "1":
        set_output_log_utf8()
    elif not (output_log_utf8 == "" or output_log_utf8 == "0"):
        print(
            "WARNING:  invalid VV_OUTPUT_LOG_UTF8 environment variable value",
            file=sys.stderr,
        )

    default_cors_policy_mode = CorsPolicyMode.localapps

    parser = argparse.ArgumentParser(description="VOICEVOX のエンジンです。")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="接続を受け付けるホストアドレスです。",
    )
    parser.add_argument(
        "--port", type=int, default=50032, help="接続を受け付けるポート番号です。"
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "directml", "opencl"),
        default=None,
        help="音声合成に使用するデバイスです。デフォルトはcpuです。",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=None,
        help="非推奨の旧オプションです。--device cudaと同じ意味です。",
    )
    # ハイフン表記を正式名とし、既存利用者向けにアンダースコア表記も受け付ける。
    parser.add_argument(
        "--device-index",
        "--device_index",
        dest="device_index",
        type=_non_negative_int,
        default=0,
        help="音声合成に使用するデバイス番号です。デフォルトは0です。",
    )
    parser.add_argument(
        "--opencl-platform-index",
        "--opencl_platform_index",
        dest="opencl_platform_index",
        type=_non_negative_int,
        default=0,
        help="OpenCLで使用するプラットフォーム番号です。デフォルトは0です。",
    )
    parser.add_argument(
        "--experimental",
        action="append",
        choices=("soxr",),
        default=[],
        help="任意機能を明示的に有効化します。複数機能はこのオプションを繰り返し指定します。",
    )
    parser.add_argument(
        "--voicevox_dir",
        type=Path,
        default=None,
        help="VOICEVOXのディレクトリパスです。",
    )
    parser.add_argument(
        "--speaker_info_dir",
        type=Path,
        default=None,
        help="MYCOEIROINKを展開したspeaker_infoディレクトリです。",
    )
    parser.add_argument(
        "--voicelib_dir",
        type=Path,
        default=None,
        action="append",
        help="VOICEVOX COREのディレクトリパスです。",
    )
    parser.add_argument(
        "--runtime_dir",
        type=Path,
        default=None,
        action="append",
        help="VOICEVOX COREで使用するライブラリのディレクトリパスです。",
    )
    parser.add_argument(
        "--enable_mock",
        action="store_true",
        help="旧起動引数との互換用です。現行の合成経路は変更しません。",
    )
    parser.add_argument(
        "--enable_cancellable_synthesis",
        action="store_true",
        help="指定すると音声合成を途中でキャンセルできるようになります。",
    )
    parser.add_argument("--init_processes", type=int, default=2)
    parser.add_argument(
        "--load_all_models",
        action="store_true",
        help="指定すると起動時に全ての音声合成モデルを読み込みます。",
    )
    parser.add_argument(
        "--disable_mutable_api",
        action="store_true",
        default=decide_boolean_from_env("VV_DISABLE_MUTABLE_API"),
        help=(
            "設定・辞書・プリセットを変更するAPIを無効化します。"
            "未指定時は環境変数VV_DISABLE_MUTABLE_APIの1を有効、0または空文字を無効として扱います。"
        ),
    )

    # 引数へcpu_num_threadsの指定がなければ、環境変数をロールします。
    # 環境変数にもない場合は、Noneのままとします。
    # VV_CPU_NUM_THREADSが空文字列でなく数値でもない場合、エラー終了します。
    parser.add_argument(
        "--cpu_num_threads",
        type=int,
        default=os.getenv("VV_CPU_NUM_THREADS") or None,
        help=(
            "音声合成を行うスレッド数です。指定しないと、代わりに環境変数VV_CPU_NUM_THREADSの値が使われます。"
            "VV_CPU_NUM_THREADSが空文字列でなく数値でもない場合はエラー終了します。"
        ),
    )

    parser.add_argument(
        "--output_log_utf8",
        action="store_true",
        help=(
            "指定するとログ出力をUTF-8でおこないます。指定しないと、代わりに環境変数 VV_OUTPUT_LOG_UTF8 の値が使われます。"
            "VV_OUTPUT_LOG_UTF8 の値が1の場合はUTF-8で、0または空文字、値がない場合は環境によって自動的に決定されます。"
        ),
    )

    parser.add_argument(
        "--cors_policy_mode",
        type=CorsPolicyMode,
        choices=list(CorsPolicyMode),
        default=None,
        help=(
            "CORSの許可モード。allまたはlocalappsが指定できます。allはすべてを許可します。"
            "localappsはオリジン間リソース共有ポリシーを、app://.とlocalhost関連に限定します。"
            "その他のオリジンはallow_originオプションで追加できます。デフォルトはlocalapps。"
        ),
    )

    parser.add_argument(
        "--allow_origin",
        nargs="*",
        help="許可するオリジンを指定します。スペースで区切ることで複数指定できます。",
    )

    parser.add_argument(
        "--setting_file",
        type=Path,
        default=USER_SETTING_PATH,
        help="設定ファイルを指定できます。",
    )

    args = parser.parse_args()
    # 公開フラグは実験機能名だけにし、下位層には既存の具体的な実装名を渡す。
    args.resampler = "soxr-vhq" if "soxr" in args.experimental else "resampy"

    try:
        args.device = resolve_device(device=args.device, use_gpu=args.use_gpu)
    except ValueError as error:
        parser.error(str(error))
    if args.use_gpu is not None:
        print(
            "WARNING: --use_gpu is deprecated; use --device cuda instead.",
            file=sys.stderr,
        )
        # 子プロセスへは正規化済みのdeviceだけを渡し、二重指定扱いを避ける。
        args.use_gpu = None

    if args.output_log_utf8:
        set_output_log_utf8()

    cpu_num_threads: int | None = args.cpu_num_threads
    root_dir = args.voicevox_dir if args.voicevox_dir is not None else engine_root()
    speaker_info_dir = (
        args.speaker_info_dir
        if args.speaker_info_dir is not None
        else root_dir / "speaker_info"
    )

    synthesis_engines = make_synthesis_engines(
        device=args.device,
        device_index=args.device_index,
        opencl_platform_index=args.opencl_platform_index,
        resampler=args.resampler,
        voicelib_dirs=args.voicelib_dir,
        voicevox_dir=args.voicevox_dir,
        runtime_dirs=args.runtime_dir,
        cpu_num_threads=cpu_num_threads,
        speaker_info_dir=speaker_info_dir,
        enable_mock=args.enable_mock,
        load_all_models=args.load_all_models,
    )
    assert len(synthesis_engines) != 0, "音声合成エンジンがありません。"
    latest_core_version = max(synthesis_engines, key=_version_key)

    cancellable_engine = None
    if args.enable_cancellable_synthesis:
        cancellable_engine = CancellableEngine(args)

    setting_loader = SettingLoader(args.setting_file)

    settings = setting_loader.load_setting_file()

    cors_policy_mode = (
        args.cors_policy_mode
        if args.cors_policy_mode is not None
        else settings.cors_policy_mode
    )

    allow_origin = None
    if args.allow_origin is not None:
        allow_origin = args.allow_origin
    elif settings.allow_origin is not None:
        allow_origin = settings.allow_origin.split(" ")

    uvicorn.run(
        generate_app(
            synthesis_engines,
            latest_core_version,
            setting_loader,
            root_dir=root_dir,
            speaker_info_dir=speaker_info_dir,
            cors_policy_mode=cors_policy_mode,
            allow_origin=allow_origin,
            cancellable_engine=cancellable_engine,
            device=args.device,
            disable_mutable_api=args.disable_mutable_api,
        ),
        host=args.host,
        port=args.port,
    )
