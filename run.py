import argparse
import asyncio
import json
import multiprocessing
import os
import re
import sys
import traceback
import zipfile
from contextlib import asynccontextmanager
from io import TextIOWrapper
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import Annotated

import soundfile
import uvicorn
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    ModelLoadError,
    StyleNotFoundError,
    SynthesisError,
)
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from packaging.version import Version
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

from voicevox_engine import __version__
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.coeiroink_v2.catalog import OfficialSiteCatalogClient
from voicevox_engine.coeiroink_v2.dictionary import (
    set_dictionary as set_coeiroink_dictionary,
)
from voicevox_engine.coeiroink_v2.metadata import (
    AmbiguousStyleError as MetadataAmbiguousStyleError,
)
from voicevox_engine.coeiroink_v2.metadata import (
    SpeakerMetadataStore,
)
from voicevox_engine.coeiroink_v2.metadata import (
    StyleNotFoundError as MetadataStyleNotFoundError,
)
from voicevox_engine.coeiroink_v2.prosody import clear_prosody_cache
from voicevox_engine.coeiroink_v2.router import create_v2_router
from voicevox_engine.engine_manifest import EngineManifestLoader
from voicevox_engine.engine_manifest.engine_manifest import EngineManifest
from voicevox_engine.kana_parser import create_kana, parse_kana
from voicevox_engine.metas.metas_store import MetasStore, construct_lookup
from voicevox_engine.model import (
    AccentPhrase,
    AudioQuery,
    MorphableTargetInfo,
    ParseKanaBadRequest,
    ParseKanaError,
    ResourceFormat,
    Speaker,
    SpeakerInfo,
    SpeakerNotFoundError,
    SupportedDevicesInfo,
    UserDictWord,
    WordTypes,
)
from voicevox_engine.morphing import (
    get_morphable_targets,
    is_synthesis_morphing_permitted,
    synthesis_morphing,
    synthesis_morphing_parameter,
)
from voicevox_engine.part_of_speech_data import MAX_PRIORITY, MIN_PRIORITY
from voicevox_engine.preset import Preset, PresetError, PresetManager
from voicevox_engine.setting import (
    USER_SETTING_PATH,
    CorsPolicyMode,
    Setting,
    SettingLoader,
)
from voicevox_engine.synthesis_engine import SynthesisEngineBase, make_synthesis_engines
from voicevox_engine.synthesis_engine.make_synthesis_engines import resolve_device
from voicevox_engine.user_dict import (
    apply_word,
    delete_word,
    import_user_dict,
    read_dict,
    rewrite_word,
    update_dict,
)
from voicevox_engine.utility import (
    ConnectBase64WavesException,
    connect_base64_waves,
    delete_file,
    engine_root,
)
from voicevox_engine.voicevox_compat.mutable_api import (
    boolean_from_env as decide_boolean_from_env,
)
from voicevox_engine.voicevox_compat.mutable_api import mutability_guard
from voicevox_engine.voicevox_compat.resources import (
    ResourceManager,
    ResourceNotFoundError,
    add_resource_route,
    resource_value,
)
from voicevox_engine.voicevox_compat.unavailable_apis import add_unavailable_routes

USER_DICTIONARY_ERRORS = (OSError, RuntimeError, TypeError, ValueError, LookupError)


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

    app = FastAPI(
        title="COEIROINK Server OSS Edition",
        description=(
            "COEIROINK /v1 APIと、/voicevox配下のVOICEVOX互換会話音声APIを提供します。"
        ),
        version=__version__,
        lifespan=lifespan,
    )
    voicevox_router = APIRouter(
        prefix="/voicevox",
        generate_unique_id_function=lambda route: route.name,
    )
    add_unavailable_routes(voicevox_router)

    @voicevox_router.get("/", response_class=HTMLResponse, tags=["その他"])
    def voicevox_portal():
        return """
        <html><body>
        <h1>COEIROINK Engine (VOICEVOX-compatible API)</h1>
        <ul>
          <li><a href='/voicevox/setting'>設定</a></li>
          <li><a href='/docs'>API ドキュメント</a></li>
        </ul>
        </body></html>
        """

    @app.exception_handler(StyleNotFoundError)
    async def style_not_found_handler(
        request: Request, err: StyleNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(err)})

    @app.exception_handler(ModelLoadError)
    @app.exception_handler(SynthesisError)
    async def synthesis_error_handler(
        request: Request, err: RuntimeError
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(err)})

    @app.exception_handler(InvalidSynthesisParameterError)
    async def invalid_synthesis_parameter_handler(
        request: Request, err: InvalidSynthesisParameterError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(err)})

    # CORS用のヘッダを生成するミドルウェア
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

    # 許可されていないOriginを遮断するミドルウェア
    @app.middleware("http")
    async def block_origin_middleware(request: Request, call_next):
        isValidOrigin: bool = False
        if (
            "Origin" not in request.headers
            or "*" in allowed_origins
            or request.headers["Origin"] in allowed_origins
            or compiled_localhost_regex.fullmatch(request.headers["Origin"])
        ):  # Originのない純粋なリクエストの場合
            isValidOrigin = True

        if isValidOrigin:
            return await call_next(request)
        return JSONResponse(status_code=403, content={"detail": "Origin not allowed"})

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
    add_resource_route(voicevox_router, resource_manager)

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

    setting_ui_template = Jinja2Templates(directory=engine_root() / "ui_template")

    def get_engine(core_version: str | None) -> SynthesisEngineBase:
        if core_version is None:
            return synthesis_engines[latest_core_version]
        if core_version in synthesis_engines:
            return synthesis_engines[core_version]
        raise HTTPException(status_code=422, detail="不明なバージョンです")

    verify_mutability_allowed = mutability_guard(disable_mutable_api)

    def ensure_legacy_style_available(
        style_id: int, speaker_uuid: str | None = None
    ) -> None:
        """Validate legacy style IDs before text-only endpoints do work."""

        try:
            installed_speaker_uuid, _ = speaker_metadata_store.find_style(style_id)
        except MetadataAmbiguousStyleError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except MetadataStyleNotFoundError as error:
            raise HTTPException(
                status_code=422,
                detail=f"MYCOEIROINK style is not installed: {style_id}",
            ) from error
        if speaker_uuid is not None and installed_speaker_uuid != speaker_uuid:
            raise HTTPException(
                status_code=422,
                detail=(
                    "MYCOEIROINK style is not installed for "
                    f"speakerUuid {speaker_uuid}: {style_id}"
                ),
            )

    def require_supported_feature(feature_name: str, display_name: str) -> None:
        features = engine_manifest_loader.load_manifest().supported_features
        if not getattr(features, feature_name):
            raise HTTPException(
                status_code=501,
                detail=f"COEIROINKは{display_name}を提供していません。",
            )

    @voicevox_router.post(
        "/audio_query",
        response_model=AudioQuery,
        tags=["クエリ作成"],
        summary="音声合成用のクエリを作成する",
    )
    def audio_query(
        text: str,
        speaker: int,
        enable_katakana_english: bool = True,
        core_version: str | None = None,
    ):
        """
        クエリの初期値を得ます。ここで得られたクエリはそのまま音声合成に利用できます。各値の意味は`Schemas`を参照してください。
        """
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        accent_phrases = engine.create_accent_phrases(
            text,
            speaker_id=speaker,
            enable_katakana_english=enable_katakana_english,
        )
        return AudioQuery(
            accent_phrases=accent_phrases,
            speedScale=1,
            pitchScale=0,
            intonationScale=1,
            volumeScale=1,
            prePhonemeLength=0.1,
            postPhonemeLength=0.1,
            pauseLength=None,
            pauseLengthScale=1,
            outputSamplingRate=engine.default_sampling_rate,
            outputStereo=False,
            kana=create_kana(accent_phrases),
        )

    @voicevox_router.post(
        "/audio_query_from_preset",
        response_model=AudioQuery,
        tags=["クエリ作成"],
        summary="音声合成用のクエリをプリセットを用いて作成する",
    )
    def audio_query_from_preset(
        text: str,
        preset_id: int,
        enable_katakana_english: bool = True,
        core_version: str | None = None,
    ):
        """
        クエリの初期値を得ます。ここで得られたクエリはそのまま音声合成に利用できます。各値の意味は`Schemas`を参照してください。
        """
        engine = get_engine(core_version)
        try:
            presets = preset_manager.load_presets()
        except PresetError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        for preset in presets:
            if preset.id == preset_id:
                selected_preset = preset
                break
        else:
            raise HTTPException(
                status_code=422, detail="該当するプリセットIDが見つかりません"
            )

        ensure_legacy_style_available(
            selected_preset.style_id, selected_preset.speaker_uuid
        )
        accent_phrases = engine.create_accent_phrases(
            text,
            speaker_id=selected_preset.style_id,
            enable_katakana_english=enable_katakana_english,
        )
        query = AudioQuery(
            accent_phrases=accent_phrases,
            speedScale=selected_preset.speedScale,
            pitchScale=selected_preset.pitchScale,
            intonationScale=selected_preset.intonationScale,
            volumeScale=selected_preset.volumeScale,
            prePhonemeLength=selected_preset.prePhonemeLength,
            postPhonemeLength=selected_preset.postPhonemeLength,
            pauseLength=selected_preset.pauseLength,
            pauseLengthScale=selected_preset.pauseLengthScale,
            outputSamplingRate=engine.default_sampling_rate,
            outputStereo=False,
            kana=create_kana(accent_phrases),
        )
        return query

    @voicevox_router.post(
        "/accent_phrases",
        response_model=list[AccentPhrase],
        tags=["クエリ編集"],
        summary="テキストからアクセント句を得る",
        responses={
            400: {
                "description": "読み仮名のパースに失敗",
                "model": ParseKanaBadRequest,
            }
        },
    )
    def accent_phrases(
        text: str,
        speaker: int,
        is_kana: bool = False,
        enable_katakana_english: bool = True,
        core_version: str | None = None,
    ):
        """
        テキストからアクセント句を得ます。
        is_kanaが`true`のとき、テキストは次のようなAquesTalkライクな記法に従う読み仮名として処理されます。デフォルトは`false`です。
        * 全てのカナはカタカナで記述される
        * アクセント句は`/`または`、`で区切る。`、`で区切った場合に限り無音区間が挿入される。
        * カナの手前に`_`を入れるとそのカナは無声化される
        * アクセント位置を`'`で指定する。全てのアクセント句にはアクセント位置を1つ指定する必要がある。
        * アクセント句末に`？`(全角)を入れることにより疑問文の発音ができる。
        """
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        if is_kana:
            try:
                accent_phrases = parse_kana(text)
            except ParseKanaError as err:
                raise HTTPException(
                    status_code=400,
                    detail=ParseKanaBadRequest(err).model_dump(),
                ) from err
            return engine.replace_mora_data(
                accent_phrases=accent_phrases, speaker_id=speaker
            )

        return engine.create_accent_phrases(
            text,
            speaker_id=speaker,
            enable_katakana_english=enable_katakana_english,
        )

    @voicevox_router.post(
        "/mora_data",
        response_model=list[AccentPhrase],
        tags=["クエリ編集"],
        summary="アクセント句から音高・音素長を得る",
    )
    def mora_data(
        accent_phrases: list[AccentPhrase],
        speaker: int,
        core_version: str | None = None,
    ):
        require_supported_feature("adjust_phoneme_length", "モーラデータ調整機能")
        require_supported_feature("adjust_mora_pitch", "モーラデータ調整機能")
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        return engine.replace_mora_data(accent_phrases, speaker_id=speaker)

    @voicevox_router.post(
        "/mora_length",
        response_model=list[AccentPhrase],
        tags=["クエリ編集"],
        summary="アクセント句から音素長を得る",
    )
    def mora_length(
        accent_phrases: list[AccentPhrase],
        speaker: int,
        core_version: str | None = None,
    ):
        require_supported_feature("adjust_phoneme_length", "音素長調整機能")
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        return engine.replace_phoneme_length(
            accent_phrases=accent_phrases, speaker_id=speaker
        )

    @voicevox_router.post(
        "/mora_pitch",
        response_model=list[AccentPhrase],
        tags=["クエリ編集"],
        summary="アクセント句から音高を得る",
    )
    def mora_pitch(
        accent_phrases: list[AccentPhrase],
        speaker: int,
        core_version: str | None = None,
    ):
        require_supported_feature("adjust_mora_pitch", "モーラ音高調整機能")
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        return engine.replace_mora_pitch(
            accent_phrases=accent_phrases, speaker_id=speaker
        )

    @voicevox_router.post(
        "/synthesis",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="音声合成する",
    )
    def synthesis(
        query: AudioQuery,
        speaker: int,
        enable_interrogative_upspeak: bool = Query(
            default=True,
            description="疑問系のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        wave = engine.synthesis(
            query=query,
            speaker_id=speaker,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
        )

        with NamedTemporaryFile(delete=False) as f:
            soundfile.write(
                file=f, data=wave, samplerate=query.outputSamplingRate, format="WAV"
            )

        return FileResponse(
            f.name,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, f.name),
        )

    @voicevox_router.post(
        "/cancellable_synthesis",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="音声合成する（キャンセル可能）",
    )
    def cancellable_synthesis(
        query: AudioQuery,
        speaker: int,
        request: Request,
        enable_interrogative_upspeak: bool = Query(
            default=True,
            description="疑問系のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        ensure_legacy_style_available(speaker)
        if cancellable_engine is None:
            raise HTTPException(
                status_code=404,
                detail="実験的機能はデフォルトで無効になっています。使用するには引数を指定してください。",
            )
        f_name = cancellable_engine._synthesis_impl(
            query=query,
            speaker_id=speaker,
            request=request,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
            core_version=core_version,
        )
        if f_name == "":
            raise HTTPException(status_code=422, detail="不明なバージョンです")

        return FileResponse(
            f_name,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, f_name),
        )

    @voicevox_router.post(
        "/multi_synthesis",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
        tags=["音声合成"],
        summary="複数まとめて音声合成する",
    )
    def multi_synthesis(
        queries: list[AudioQuery],
        speaker: int,
        enable_interrogative_upspeak: bool = Query(
            default=True,
            description="疑問系のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        """同一話者・サンプリングレートの複数クエリを順に合成し、一時ZIPとして応答する。"""

        if not queries:
            raise HTTPException(
                status_code=422,
                detail="音声合成クエリが1件もありません。",
            )
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        sampling_rate = queries[0].outputSamplingRate
        if any(query.outputSamplingRate != sampling_rate for query in queries[1:]):
            raise HTTPException(
                status_code=422, detail="サンプリングレートが異なるクエリがあります"
            )

        with NamedTemporaryFile(delete=False) as f:
            zip_path = f.name
        try:
            with zipfile.ZipFile(zip_path, mode="a") as zip_file:
                for i in range(len(queries)):
                    with TemporaryFile() as wav_file:
                        wave = engine.synthesis(
                            query=queries[i],
                            speaker_id=speaker,
                            enable_interrogative_upspeak=enable_interrogative_upspeak,
                        )
                        soundfile.write(
                            file=wav_file,
                            data=wave,
                            samplerate=sampling_rate,
                            format="WAV",
                        )
                        wav_file.seek(0)
                        zip_file.writestr(f"{str(i + 1).zfill(3)}.wav", wav_file.read())
        except Exception:
            delete_file(zip_path)
            raise

        return FileResponse(
            zip_path,
            media_type="application/zip",
            background=BackgroundTask(delete_file, zip_path),
        )

    @voicevox_router.post(
        "/morphable_targets",
        response_model=list[dict[str, MorphableTargetInfo]],
        tags=["音声合成"],
        summary="指定した話者に対してエンジン内の話者がモーフィングが可能か判定する",
    )
    def morphable_targets(
        base_speakers: list[int],
        core_version: str | None = None,
    ):
        """
        指定されたベース話者に対してエンジン内の各話者がモーフィング機能を利用可能か返します。
        モーフィングの許可/禁止は`/voicevox/speakers`の`speaker.supported_features.synthesis_morphing`に記載されています。
        プロパティが存在しない場合は、モーフィングが許可されているとみなします。
        返り値の話者はstring型なので注意。
        """
        require_supported_feature("synthesis_morphing", "音声モーフィング機能")
        engine = get_engine(core_version)

        try:
            speakers = metas_store.load_combined_metas(engine=engine)
            morphable_targets = get_morphable_targets(
                speakers=speakers, base_speakers=base_speakers
            )
            # jsonはint型のキーを持てないので、string型に変換する
            return [
                {str(k): v for k, v in morphable_target.items()}
                for morphable_target in morphable_targets
            ]
        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"該当する話者(speaker={e.speaker})が見つかりません",
            ) from e

    @voicevox_router.post(
        "/synthesis_morphing",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["音声合成"],
        summary="2人の話者でモーフィングした音声を合成する",
    )
    def _synthesis_morphing(
        query: AudioQuery,
        base_speaker: int,
        target_speaker: int,
        morph_rate: float = Query(..., ge=0.0, le=1.0),
        enable_interrogative_upspeak: bool = Query(
            default=True,
            description="疑問系のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        """
        指定された2人の話者で音声を合成、指定した割合でモーフィングした音声を得ます。
        モーフィングの割合は`morph_rate`で指定でき、0.0でベースの話者、1.0でターゲットの話者に近づきます。
        """
        require_supported_feature("synthesis_morphing", "音声モーフィング機能")
        engine = get_engine(core_version)

        try:
            speakers = metas_store.load_combined_metas(engine=engine)
            speaker_lookup = construct_lookup(speakers=speakers)
            is_permitted = is_synthesis_morphing_permitted(
                speaker_lookup, base_speaker, target_speaker
            )
            if not is_permitted:
                raise HTTPException(
                    status_code=400,
                    detail="指定された話者ペアでのモーフィングはできません",
                )
        except SpeakerNotFoundError as e:
            raise HTTPException(
                status_code=404,
                detail=f"該当する話者(speaker={e.speaker})が見つかりません",
            ) from e

        morph_param = synthesis_morphing_parameter(
            engine=engine,
            query=query,
            base_speaker=base_speaker,
            target_speaker=target_speaker,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
        )

        morph_wave = synthesis_morphing(
            morph_param=morph_param,
            morph_rate=morph_rate,
            output_stereo=query.outputStereo,
        )

        with NamedTemporaryFile(delete=False) as f:
            soundfile.write(
                file=f,
                data=morph_wave,
                samplerate=morph_param.fs,
                format="WAV",
            )

        return FileResponse(
            f.name,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, f.name),
        )

    @voicevox_router.post(
        "/connect_waves",
        response_class=FileResponse,
        responses={
            200: {
                "content": {
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
        tags=["その他"],
        summary="base64エンコードされた複数のwavデータを一つに結合する",
    )
    def connect_waves(waves: list[str]):
        """
        base64エンコードされたwavデータを一纏めにし、wavファイルで返します。
        """
        try:
            waves_nparray, sampling_rate = connect_base64_waves(waves)
        except ConnectBase64WavesException as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        with NamedTemporaryFile(delete=False) as f:
            soundfile.write(
                file=f,
                data=waves_nparray,
                samplerate=sampling_rate,
                format="WAV",
            )

        return FileResponse(
            f.name,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, f.name),
        )

    @voicevox_router.post(
        "/validate_kana",
        response_model=bool,
        tags=["その他"],
        summary="テキストがAquesTalkライクな記法に従っているか判定する",
        responses={
            400: {
                "description": "テキストが不正です",
                "model": ParseKanaBadRequest,
            }
        },
    )
    def validate_kana(text: str):
        try:
            parse_kana(text)
            return True
        except ParseKanaError as err:
            raise HTTPException(
                status_code=400,
                detail=ParseKanaBadRequest(err).model_dump(),
            ) from err

    @voicevox_router.get("/presets", response_model=list[Preset], tags=["その他"])
    def get_presets():
        """
        エンジンが保持しているプリセットの設定を返します

        Returns
        -------
        presets: List[Preset]
            プリセットのリスト
        """
        try:
            presets = preset_manager.load_presets()
        except PresetError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return presets

    @voicevox_router.post(
        "/add_preset",
        response_model=int,
        tags=["その他"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def add_preset(preset: Preset):
        """
        新しいプリセットを追加します

        Parameters
        -------
        preset: Preset
            新しいプリセット。
            プリセットIDが既存のものと重複している場合は、新規のプリセットIDが採番されます。

        Returns
        -------
        id: int
            追加したプリセットのプリセットID
        """
        try:
            id = preset_manager.add_preset(preset)
        except PresetError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return id

    @voicevox_router.post(
        "/update_preset",
        response_model=int,
        tags=["その他"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def update_preset(preset: Preset):
        """
        既存のプリセットを更新します

        Parameters
        -------
        preset: Preset
            更新するプリセット。
            プリセットIDが更新対象と一致している必要があります。

        Returns
        -------
        id: int
            更新したプリセットのプリセットID
        """
        try:
            id = preset_manager.update_preset(preset)
        except PresetError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return id

    @voicevox_router.post(
        "/delete_preset",
        status_code=204,
        tags=["その他"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def delete_preset(id: int):
        """
        既存のプリセットを削除します

        Parameters
        -------
        id: int
            削除するプリセットのプリセットID

        """
        try:
            preset_manager.delete_preset(id)
        except PresetError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err
        return Response(status_code=204)

    @voicevox_router.get("/version", tags=["その他"])
    def version() -> str:
        return __version__

    @voicevox_router.get("/core_versions", response_model=list[str], tags=["その他"])
    def core_versions() -> list[str]:
        return Response(
            content=json.dumps(list(synthesis_engines.keys())),
            media_type="application/json",
        )

    @voicevox_router.get("/speakers", response_model=list[Speaker], tags=["その他"])
    def speakers(
        core_version: str | None = None,
    ):
        engine = get_engine(core_version)
        return metas_store.load_combined_metas(engine=engine)

    @voicevox_router.get("/speaker_info", response_model=SpeakerInfo, tags=["その他"])
    def speaker_info(
        request: Request,
        speaker_uuid: str,
        resource_format: ResourceFormat = ResourceFormat.BASE64,
        core_version: str | None = None,
    ):
        """
        指定されたspeaker_uuidに関する情報をjson形式で返します。
        画像や音声はresource_formatで指定した形式で返されます。

        Returns
        -------
        ret_data: SpeakerInfo
        """

        def resolve_resource(path: Path) -> str:
            return resource_value(
                resource_manager,
                request,
                resource_format,
                path,
            )

        speakers = json.loads(get_engine(core_version).speakers)
        for i in range(len(speakers)):
            if speakers[i]["speaker_uuid"] == speaker_uuid:
                speaker = speakers[i]
                break
        else:
            raise HTTPException(status_code=404, detail="該当する話者が見つかりません")

        speaker_dir = metas_store.speaker_path(speaker_uuid)
        try:
            policy = (speaker_dir / "policy.md").read_text("utf-8")
            portrait = resolve_resource(speaker_dir / "portrait.png")
            style_infos = []
            for style in speaker["styles"]:
                id = style["id"]
                icon = resolve_resource(speaker_dir / f"icons/{id}.png")
                style_portrait_path = speaker_dir / f"portraits/{id}.png"
                style_portrait = (
                    resolve_resource(style_portrait_path)
                    if style_portrait_path.exists()
                    else None
                )
                # v2メタデータ境界と同じ列挙規則を使い、サンプル数や番号の欠番を固定値で仮定しない。
                voice_samples = [
                    resolve_resource(path)
                    for path in speaker_metadata_store.voice_sample_paths(
                        speaker_uuid,
                        id,
                    )
                ]
                style_infos.append(
                    {
                        "id": id,
                        "icon": icon,
                        "portrait": style_portrait,
                        "voice_samples": voice_samples,
                    }
                )
        except (FileNotFoundError, ResourceNotFoundError) as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail="追加情報が見つかりませんでした"
            ) from error

        return {"policy": policy, "portrait": portrait, "style_infos": style_infos}

    @voicevox_router.post("/initialize_speaker", status_code=204, tags=["その他"])
    def initialize_speaker(
        speaker: int,
        skip_reinit: bool = Query(
            False, description="既に初期化済みの話者の再初期化をスキップするかどうか"
        ),
        core_version: str | None = None,
    ):
        """
        指定されたspeaker_idの話者を初期化します。
        実行しなくても他のAPIは使用できますが、初回実行時に時間がかかることがあります。
        """
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        engine.initialize_speaker_synthesis(speaker_id=speaker, skip_reinit=skip_reinit)
        return Response(status_code=204)

    @voicevox_router.get(
        "/is_initialized_speaker", response_model=bool, tags=["その他"]
    )
    def is_initialized_speaker(speaker: int, core_version: str | None = None):
        """
        指定されたspeaker_idの話者が初期化されているかどうかを返します。
        """
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        return engine.is_initialized_speaker_synthesis(speaker)

    @voicevox_router.get(
        "/user_dict", response_model=dict[str, UserDictWord], tags=["ユーザー辞書"]
    )
    def get_user_dict_words():
        """
        ユーザー辞書に登録されている単語の一覧を返します。
        単語の表層形(surface)は正規化済みの物を返します。

        Returns
        -------
        Dict[str, UserDictWord]
            単語のUUIDとその詳細
        """
        try:
            return read_dict()
        except USER_DICTIONARY_ERRORS as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=422, detail="辞書の読み込みに失敗しました。"
            ) from error

    @voicevox_router.post(
        "/user_dict_word",
        response_model=str,
        tags=["ユーザー辞書"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def add_user_dict_word(
        surface: str,
        pronunciation: str,
        accent_type: int,
        word_type: WordTypes | None = None,
        priority: Annotated[
            int | None,
            Query(ge=MIN_PRIORITY, le=MAX_PRIORITY),
        ] = None,
    ):
        """
        ユーザー辞書に言葉を追加します。

        Parameters
        ----------
        surface : str
            言葉の表層形
        pronunciation: str
            言葉の発音（カタカナ）
        accent_type: int
            アクセント型（音が下がる場所を指す）
        word_type: WordTypes, optional
            PROPER_NOUN（固有名詞）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（語尾）のいずれか
        priority: int, optional
            単語の優先度（0から10までの整数）
            数字が大きいほど優先度が高くなる
            1から9までの値を指定することを推奨
        """
        try:
            word_uuid = apply_word(
                surface=surface,
                pronunciation=pronunciation,
                accent_type=accent_type,
                word_type=word_type,
                priority=priority,
            )
            clear_prosody_cache()
            return word_uuid
        except ValidationError as e:
            raise HTTPException(
                status_code=422, detail="パラメータに誤りがあります。\n" + str(e)
            ) from e
        except HTTPException:
            raise
        except USER_DICTIONARY_ERRORS as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=422, detail="ユーザー辞書への追加に失敗しました。"
            ) from error

    @voicevox_router.put(
        "/user_dict_word/{word_uuid}",
        status_code=204,
        tags=["ユーザー辞書"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def rewrite_user_dict_word(
        surface: str,
        pronunciation: str,
        accent_type: int,
        word_uuid: str,
        word_type: WordTypes | None = None,
        priority: Annotated[
            int | None,
            Query(ge=MIN_PRIORITY, le=MAX_PRIORITY),
        ] = None,
    ):
        """
        ユーザー辞書に登録されている言葉を更新します。

        Parameters
        ----------
        surface : str
            言葉の表層形
        pronunciation: str
            言葉の発音（カタカナ）
        accent_type: int
            アクセント型（音が下がる場所を指す）
        word_uuid: str
            更新する言葉のUUID
        word_type: WordTypes, optional
            PROPER_NOUN（固有名詞）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（語尾）のいずれか
        priority: int, optional
            単語の優先度（0から10までの整数）
            数字が大きいほど優先度が高くなる
            1から9までの値を指定することを推奨
        """
        try:
            rewrite_word(
                surface=surface,
                pronunciation=pronunciation,
                accent_type=accent_type,
                word_uuid=word_uuid,
                word_type=word_type,
                priority=priority,
            )
            clear_prosody_cache()
            return Response(status_code=204)
        except HTTPException:
            raise
        except ValidationError as e:
            raise HTTPException(
                status_code=422, detail="パラメータに誤りがあります。\n" + str(e)
            ) from e
        except USER_DICTIONARY_ERRORS as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=422, detail="ユーザー辞書の更新に失敗しました。"
            ) from error

    @voicevox_router.delete(
        "/user_dict_word/{word_uuid}",
        status_code=204,
        tags=["ユーザー辞書"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def delete_user_dict_word(word_uuid: str):
        """
        ユーザー辞書に登録されている言葉を削除します。

        Parameters
        ----------
        word_uuid: str
            削除する言葉のUUID
        """
        try:
            delete_word(word_uuid=word_uuid)
            clear_prosody_cache()
            return Response(status_code=204)
        except HTTPException:
            raise
        except USER_DICTIONARY_ERRORS as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=422, detail="ユーザー辞書の更新に失敗しました。"
            ) from error

    @voicevox_router.post(
        "/import_user_dict",
        status_code=204,
        tags=["ユーザー辞書"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def import_user_dict_words(
        import_dict_data: dict[str, UserDictWord], override: bool
    ):
        """
        他のユーザー辞書をインポートします。

        Parameters
        ----------
        import_dict_data: Dict[str, UserDictWord]
            インポートするユーザー辞書のデータ
        override: bool
            重複したエントリがあった場合、上書きするかどうか
        """
        try:
            import_user_dict(dict_data=import_dict_data, override=override)
            clear_prosody_cache()
            return Response(status_code=204)
        except HTTPException:
            raise
        except USER_DICTIONARY_ERRORS as error:
            traceback.print_exc()
            raise HTTPException(
                status_code=422, detail="ユーザー辞書のインポートに失敗しました。"
            ) from error

    @voicevox_router.get(
        "/supported_devices", response_model=SupportedDevicesInfo, tags=["その他"]
    )
    def supported_devices(
        core_version: str | None = None,
    ):
        supported_devices = get_engine(core_version).supported_devices
        if supported_devices is None:
            raise HTTPException(status_code=422, detail="非対応の機能です。")
        device_info = json.loads(supported_devices)
        device_info.setdefault("dml", False)
        # Core内部のOpenCL拡張はVOICEVOX互換schemaへ混ぜず、既存3項目だけを返す。
        return SupportedDevicesInfo(
            cpu=device_info["cpu"],
            cuda=device_info["cuda"],
            dml=device_info["dml"],
        )

    @voicevox_router.get(
        "/engine_manifest", response_model=EngineManifest, tags=["その他"]
    )
    def engine_manifest():
        return engine_manifest_loader.load_manifest()

    @voicevox_router.get("/setting", response_class=HTMLResponse, tags=["設定"])
    def setting_get(request: Request):
        settings = setting_loader.load_setting_file()

        cors_policy_mode = settings.cors_policy_mode
        allow_origin = settings.allow_origin

        if allow_origin is None:
            allow_origin = ""

        return setting_ui_template.TemplateResponse(
            request=request,
            name="ui.html",
            context={
                "cors_policy_mode": cors_policy_mode,
                "allow_origin": allow_origin,
            },
        )

    @voicevox_router.post(
        "/setting",
        status_code=204,
        tags=["設定"],
        dependencies=[Depends(verify_mutability_allowed)],
    )
    def setting_post(
        cors_policy_mode: Annotated[CorsPolicyMode, Form()],
        allow_origin: Annotated[str | None, Form()] = None,
    ):
        settings = Setting(
            cors_policy_mode=cors_policy_mode,
            allow_origin=allow_origin,
        )

        # 更新した設定へ上書き
        setting_loader.dump_setting_file(settings)

        return Response(status_code=204)

    app.include_router(voicevox_router)
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
