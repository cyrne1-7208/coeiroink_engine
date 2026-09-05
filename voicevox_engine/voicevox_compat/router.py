"""VOICEVOX互換APIのルート群。"""

import json
import traceback
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryFile
from typing import Annotated

import soundfile
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from starlette.background import BackgroundTask

from voicevox_engine import __version__
from voicevox_engine.cancellable_engine import CancellableEngine
from voicevox_engine.coeiroink_v2.metadata import (
    AmbiguousStyleError as MetadataAmbiguousStyleError,
)
from voicevox_engine.coeiroink_v2.metadata import SpeakerMetadataStore
from voicevox_engine.coeiroink_v2.metadata import (
    SpeakerNotFoundError as MetadataSpeakerNotFoundError,
)
from voicevox_engine.coeiroink_v2.metadata import (
    StyleNotFoundError as MetadataStyleNotFoundError,
)
from voicevox_engine.coeiroink_v2.prosody import clear_prosody_cache
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
from voicevox_engine.setting import CorsPolicyMode, Setting, SettingLoader
from voicevox_engine.synthesis_engine import SynthesisEngineBase
from voicevox_engine.user_dict import (
    apply_word,
    delete_word,
    import_user_dict,
    read_dict,
    rewrite_word,
)
from voicevox_engine.utility import (
    ConnectBase64WavesException,
    connect_base64_waves,
    delete_file,
    engine_root,
)
from voicevox_engine.voicevox_compat.resources import (
    ResourceManager,
    ResourceNotFoundError,
    add_resource_route,
    resource_value,
)
from voicevox_engine.voicevox_compat.unavailable_apis import add_unavailable_routes

USER_DICTIONARY_ERRORS = (OSError, RuntimeError, TypeError, ValueError, LookupError)


def _wave_file_response(wave, sampling_rate: int) -> FileResponse:
    """WAV一時ファイルを応答へ移譲し、移譲前の失敗時だけ直ちに削除する。"""

    path: str | None = None
    try:
        with NamedTemporaryFile(delete=False) as file:
            path = file.name
            soundfile.write(
                file=file,
                data=wave,
                samplerate=sampling_rate,
                format="WAV",
            )
        return FileResponse(
            path,
            media_type="audio/wav",
            background=BackgroundTask(delete_file, path),
        )
    except Exception:
        if path is not None:
            delete_file(path)
        raise


@dataclass(slots=True)
class VoicevoxRouterDependencies:
    """互換ルート群が共有するEngineの既存マネージャー。"""

    synthesis_engines: dict[str, SynthesisEngineBase]
    latest_core_version: str
    preset_manager: PresetManager
    engine_manifest_loader: EngineManifestLoader
    metas_store: MetasStore
    speaker_metadata_store: SpeakerMetadataStore
    resource_manager: ResourceManager
    setting_loader: SettingLoader
    cancellable_engine: CancellableEngine | None
    verify_mutability_allowed: Callable[[], None]

    def get_engine(self, core_version: str | None) -> SynthesisEngineBase:
        if core_version is None:
            return self.synthesis_engines[self.latest_core_version]
        if core_version in self.synthesis_engines:
            return self.synthesis_engines[core_version]
        raise HTTPException(status_code=422, detail="不明なバージョンです")

    def ensure_legacy_style_available(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> None:
        """テキスト解析前に従来形式のスタイルIDを検証する。"""

        try:
            self.speaker_metadata_store.find_style(style_id, speaker_uuid)
        except MetadataAmbiguousStyleError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (MetadataSpeakerNotFoundError, MetadataStyleNotFoundError) as error:
            detail = (
                f"MYCOEIROINK style is not installed: {style_id}"
                if speaker_uuid is None
                else (
                    "MYCOEIROINK style is not installed for "
                    f"speakerUuid {speaker_uuid}: {style_id}"
                )
            )
            raise HTTPException(
                status_code=422,
                detail=detail,
            ) from error

    def require_supported_feature(self, feature_name: str, display_name: str) -> None:
        features = self.engine_manifest_loader.load_manifest().supported_features
        if not getattr(features, feature_name):
            raise HTTPException(
                status_code=501,
                detail=f"COEIROINKは{display_name}を提供していません。",
            )


def _add_portal_route(
    voicevox_router: APIRouter,
) -> None:
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


def _add_tts_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    preset_manager = dependencies.preset_manager
    metas_store = dependencies.metas_store
    cancellable_engine = dependencies.cancellable_engine
    get_engine = dependencies.get_engine
    ensure_legacy_style_available = dependencies.ensure_legacy_style_available
    require_supported_feature = dependencies.require_supported_feature

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

        # VOICEVOXのAudioQueryとsynthesisは話者UUIDを運べないため、重複style IDは使えないクエリを返す前に拒否する。
        ensure_legacy_style_available(selected_preset.style_id)
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
                "description": "読み仮名のパースに失敗した場合",
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
        description="Engineマニフェストで必要な調整機能が未対応の場合は501を返します。",
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
        description="Engineマニフェストで音素長調整が未対応の場合は501を返します。",
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
        description="Engineマニフェストでモーラ音高調整が未対応の場合は501を返します。",
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
            default=False,
            description="疑問形のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        if enable_interrogative_upspeak:
            require_supported_feature("interrogative_upspeak", "疑問文の自動調整機能")
        ensure_legacy_style_available(speaker)
        engine = get_engine(core_version)
        wave = engine.synthesis(
            query=query,
            speaker_id=speaker,
            enable_interrogative_upspeak=enable_interrogative_upspeak,
        )

        return _wave_file_response(wave, query.outputSamplingRate)

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
            default=False,
            description="疑問形のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        if enable_interrogative_upspeak:
            require_supported_feature("interrogative_upspeak", "疑問文の自動調整機能")
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

        try:
            return FileResponse(
                f_name,
                media_type="audio/wav",
                background=BackgroundTask(delete_file, f_name),
            )
        except Exception:
            delete_file(f_name)
            raise

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
            default=False,
            description="疑問形のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        """同一話者・サンプリングレートの複数クエリを順に合成し、一時ZIPとして応答する。"""

        if enable_interrogative_upspeak:
            require_supported_feature("interrogative_upspeak", "疑問文の自動調整機能")
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
        summary="指定した話者に対してエンジン内の話者がモーフィング可能か判定する",
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
            default=False,
            description="疑問形のテキストが与えられたら語尾を自動調整する",
        ),
        core_version: str | None = None,
    ):
        """
        指定された2人の話者で音声を合成し、指定した割合でモーフィングした音声を得ます。
        モーフィングの割合は`morph_rate`で指定でき、0.0でベースの話者、1.0でターゲットの話者に近づきます。
        """
        require_supported_feature("synthesis_morphing", "音声モーフィング機能")
        if enable_interrogative_upspeak:
            require_supported_feature("interrogative_upspeak", "疑問文の自動調整機能")
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

        return _wave_file_response(morph_wave, int(morph_param.fs))

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
        base64エンコードされた複数のwavデータを1つに結合し、wavファイルとして返します。
        """
        try:
            waves_nparray, sampling_rate = connect_base64_waves(waves)
        except ConnectBase64WavesException as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

        return _wave_file_response(waves_nparray, sampling_rate)

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


def _add_preset_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    preset_manager = dependencies.preset_manager
    verify_mutability_allowed = dependencies.verify_mutability_allowed

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


def _add_character_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    synthesis_engines = dependencies.synthesis_engines
    metas_store = dependencies.metas_store
    speaker_metadata_store = dependencies.speaker_metadata_store
    resource_manager = dependencies.resource_manager
    get_engine = dependencies.get_engine
    ensure_legacy_style_available = dependencies.ensure_legacy_style_available

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

        speakers = metas_store.load_combined_metas(engine=get_engine(core_version))
        speaker = next(
            (item for item in speakers if item.speaker_uuid == speaker_uuid),
            None,
        )
        if speaker is None:
            raise HTTPException(status_code=404, detail="該当する話者が見つかりません")

        speaker_dir = metas_store.speaker_path(speaker_uuid)
        try:
            policy = (speaker_dir / "policy.md").read_text("utf-8")
            portrait = resolve_resource(speaker_dir / "portrait.png")
            style_infos = []
            for style in speaker.styles:
                id = style.id
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


def _add_user_dictionary_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    verify_mutability_allowed = dependencies.verify_mutability_allowed

    @voicevox_router.get(
        "/user_dict", response_model=dict[str, UserDictWord], tags=["ユーザー辞書"]
    )
    def get_user_dict_words():
        """
        ユーザー辞書に登録されている単語の一覧を返します。
        単語の表層形（surface）は正規化済みの形式で返します。

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
        ユーザー辞書に単語を追加します。

        Parameters
        ----------
        surface : str
            単語の表層形
        pronunciation: str
            単語の発音（カタカナ）
        accent_type: int
            アクセント型（音が下がる場所を指す）
        word_type: WordTypes, optional
            PROPER_NOUN（固有名詞）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（接尾辞）のいずれか
        priority: int, optional
            単語の優先度（0から10までの整数）
            数値が大きいほど優先度が高くなります（1から9までの値を推奨）。
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
        ユーザー辞書に登録されている単語を更新します。

        Parameters
        ----------
        surface : str
            単語の表層形
        pronunciation: str
            単語の発音（カタカナ）
        accent_type: int
            アクセント型（音が下がる場所を指す）
        word_uuid: str
            更新する単語のUUID
        word_type: WordTypes, optional
            PROPER_NOUN（固有名詞）、COMMON_NOUN（普通名詞）、VERB（動詞）、ADJECTIVE（形容詞）、SUFFIX（接尾辞）のいずれか
        priority: int, optional
            単語の優先度（0から10までの整数）
            数値が大きいほど優先度が高くなります（1から9までの値を推奨）。
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
        ユーザー辞書に登録されている単語を削除します。

        Parameters
        ----------
        word_uuid: str
            削除する単語のUUID
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
                status_code=422, detail="ユーザー辞書からの削除に失敗しました。"
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


def _add_engine_info_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    engine_manifest_loader = dependencies.engine_manifest_loader
    get_engine = dependencies.get_engine

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


def _add_setting_routes(
    voicevox_router: APIRouter, dependencies: VoicevoxRouterDependencies
) -> None:
    setting_loader = dependencies.setting_loader
    verify_mutability_allowed = dependencies.verify_mutability_allowed
    setting_ui_template = Jinja2Templates(directory=engine_root() / "ui_template")

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

        # 更新した設定で上書き
        setting_loader.dump_setting_file(settings)

        return Response(status_code=204)


def create_voicevox_router(
    dependencies: VoicevoxRouterDependencies,
) -> APIRouter:
    """既存のCOEIROINK合成器へ委譲するVOICEVOX互換ルーターを構築する。"""

    router = APIRouter(
        prefix="/voicevox",
        generate_unique_id_function=lambda route: route.name,
    )
    add_unavailable_routes(router)
    add_resource_route(router, dependencies.resource_manager)

    # 上流VOICEVOXと同様、機能群ごとにルートを登録してアプリ組み立てからHTTP実装を分離する。
    _add_portal_route(router)
    _add_tts_routes(router, dependencies)
    _add_preset_routes(router, dependencies)
    _add_character_routes(router, dependencies)
    _add_user_dictionary_routes(router, dependencies)
    _add_engine_info_routes(router, dependencies)
    _add_setting_routes(router, dependencies)
    return router


__all__ = ["VoicevoxRouterDependencies", "create_voicevox_router"]
