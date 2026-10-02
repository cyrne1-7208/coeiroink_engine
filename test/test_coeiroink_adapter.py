"""本番アダプターのCoreへの変換と、起動設定の受け渡しを検証する。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from coeirocore import __version__ as core_version
from coeirocore.devices import DeviceBackend

from voicevox_engine.kana_parser import parse_kana
from voicevox_engine.model import AudioQuery
from voicevox_engine.synthesis_engine import CoeiroinkVoicevoxAdapter
from voicevox_engine.synthesis_engine.make_synthesis_engines import (
    make_audio_manager,
    make_synthesis_engines,
    resolve_device,
)


def test_adapter_converts_query_to_native_tokens_and_controls():
    manager = Mock(fs=44100, device="cpu")
    manager.synthesis.return_value = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    engine = CoeiroinkVoicevoxAdapter(speakers="[]", audio_manager=manager)
    query = AudioQuery(
        accent_phrases=parse_kana("カ'、_ス'？"),
        speedScale=1.25,
        pitchScale=0.2,
        intonationScale=0.8,
        volumeScale=0.5,
        prePhonemeLength=0.2,
        postPhonemeLength=0.3,
        outputSamplingRate=24000,
        outputStereo=True,
        pauseLength=0.4,
        pauseLengthScale=1.2,
    )
    before = query.model_dump()
    wave = engine.synthesis(query, speaker_id=1)
    manager.synthesis.assert_called_once_with(
        text=["^", "k", "a", "_", "s", "u", "?"],
        style_id=1,
        speed_scale=1.25,
        pitch_scale=0.2,
        intonation_scale=0.8,
        volume_scale=0.5,
        pre_phoneme_length=0.2,
        post_phoneme_length=0.3,
        output_sampling_rate=24000,
        pause_length=0.4,
        pause_length_scale=1.2,
    )
    np.testing.assert_array_equal(wave[:, 0], manager.synthesis.return_value)
    np.testing.assert_array_equal(wave[:, 1], wave[:, 0])
    assert query.model_dump() == before


def test_make_audio_manager_passes_complete_device_selection():
    with patch(
        "voicevox_engine.synthesis_engine.make_synthesis_engines.AudioManager"
    ) as factory:
        make_audio_manager(
            speaker_info_dir=Path("speaker_info"),
            device="opencl",
            device_index=2,
            opencl_platform_index=1,
            max_loaded_models=3,
            voice_smoothing=True,
        )
    factory.assert_called_once_with(
        fs=44100,
        device="opencl",
        device_index=2,
        opencl_platform_index=1,
        use_gpu=None,
        speaker_info_dir=Path("speaker_info").resolve(),
        cpu_num_threads=0,
        resampler="resampy",
        max_loaded_models=3,
        generator_only=True,
        voice_smoothing=True,
    )


def test_factory_reuses_injected_core_metadata_device_and_model_residency():
    manager = Mock(fs=44100, device="opencl")
    manager.meta_manager.get_metas_dict.return_value = []
    capabilities = {
        backend: SimpleNamespace(
            available=backend in (DeviceBackend.CPU, DeviceBackend.OPENCL)
        )
        for backend in DeviceBackend
    }
    with patch(
        "voicevox_engine.synthesis_engine.make_synthesis_engines.get_supported_device_capabilities",
        return_value=capabilities,
    ):
        engines = make_synthesis_engines(audio_manager=manager, max_loaded_models=None)
    assert list(engines) == [core_version]
    engine = engines[core_version]
    assert engine.audio_manager is manager
    assert engine.device == "opencl"
    assert json.loads(engine.speakers) == []
    assert json.loads(engine.supported_devices) == {
        "cpu": True,
        "cuda": False,
        "dml": False,
        "opencl": True,
    }
    manager.initialize_all_speakers.assert_called_once_with()


def test_resolve_device_keeps_devices_and_maps_legacy_gpu_flag():
    assert resolve_device("directml") == "directml"
    assert resolve_device("opencl") == "opencl"
    assert resolve_device(use_gpu=True) == "cuda"
    assert resolve_device(use_gpu=False) == "cpu"
    with pytest.raises(ValueError, match="同時に指定"):
        resolve_device("cpu", use_gpu=True)
