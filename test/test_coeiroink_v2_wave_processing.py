# COEIROINK v2の波形後処理パイプラインの適用順序・分岐・WORLD F0・上限検証のテスト

from __future__ import annotations

import numpy as np
import pytest
from coeirocore.pyworld_compat import load_pyworld

from voicevox_engine.coeiroink_v2 import audio as audio_helpers
from voicevox_engine.coeiroink_v2 import wave_processing
from voicevox_engine.coeiroink_v2.wave_processing import (
    WaveProcessingOptions,
    as_waveform,
    normalize_processing_algorithm,
    process_wave,
)


def _options(**overrides: object) -> WaveProcessingOptions:
    values: dict[str, object] = {
        "volume_scale": 1.0,
        "pitch_scale": 0.0,
        "intonation_scale": 1.0,
        "pre_phoneme_length": 0.0,
        "post_phoneme_length": 0.0,
        "output_sampling_rate": 16000,
        "start_trim_buffer": 0.0,
        "end_trim_buffer": 0.0,
    }
    values.update(overrides)
    return WaveProcessingOptions(**values)


def _internal_pause_durations() -> list[dict[str, object]]:
    return [
        {
            "mora": mora,
            "hira": "",
            "phonemePitches": [],
            "wavRange": {"start": start, "end": end},
        }
        for mora, start, end in (("a", 0, 2), ("pau", 2, 4), ("a", 4, 8))
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "td-psola"),
        ("coeiroink", "td-psola"),
        ("TD-PSOLA", "td-psola"),
        ("world", "world"),
        ("resampling", "resampling"),
    ],
)
def test_normalize_processing_algorithm_maps_aliases(
    raw: str | None, expected: str
) -> None:
    assert normalize_processing_algorithm(raw) == expected


def test_normalize_processing_algorithm_rejects_unknown_value() -> None:
    with pytest.raises(
        audio_helpers.AudioValidationError, match="processing_algorithm"
    ):
        normalize_processing_algorithm("unknown")


def test_as_waveform_rejects_empty_or_non_finite_values() -> None:
    with pytest.raises(audio_helpers.AudioValidationError, match="invalid waveform"):
        as_waveform([])
    with pytest.raises(audio_helpers.AudioValidationError, match="invalid waveform"):
        as_waveform(np.array([0.0, np.nan], dtype=np.float32))


def test_identity_options_skip_pitch_and_resampling(
    v2_fake_audio_manager_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = v2_fake_audio_manager_cls()
    wave = np.linspace(-0.25, 0.25, 8, dtype=np.float32)
    monkeypatch.setattr(
        wave_processing,
        "process_td_psola",
        lambda *_a, **_kw: pytest.fail("pitch processing should be skipped"),
    )
    monkeypatch.setattr(
        manager,
        "resample_output",
        lambda *_a, **_kw: pytest.fail("resampling should be skipped"),
    )

    result, rate = process_wave(manager, wave, 16000, _options())

    assert rate == 16000
    np.testing.assert_array_equal(result, wave)


def test_td_psola_and_resampling_algorithms_dispatch_to_expected_backend(
    v2_fake_audio_manager_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = v2_fake_audio_manager_cls()
    wave = np.array([-0.25, 0.0, 0.25, 0.5], dtype=np.float32)
    calls: list[str] = []

    def fake_td_psola(
        current: np.ndarray, *_args: object, **_kwargs: object
    ) -> np.ndarray:
        calls.append("td-psola")
        return current + 1.0

    def fake_pitch_shift(
        current: np.ndarray, *_args: object, **_kwargs: object
    ) -> np.ndarray:
        calls.append("resampling")
        return current + 2.0

    monkeypatch.setattr(wave_processing, "process_td_psola", fake_td_psola)
    monkeypatch.setattr(audio_helpers, "pitch_shift_resampling", fake_pitch_shift)

    td_result, _ = process_wave(
        manager,
        wave,
        16000,
        _options(processing_algorithm="coeiroink", pitch_scale=0.2),
    )
    np.testing.assert_array_equal(td_result, wave + 1.0)
    assert calls == ["td-psola"]

    calls.clear()
    rs_result, _ = process_wave(
        manager,
        wave,
        16000,
        _options(
            processing_algorithm="resampling",
            pitch_scale=0.2,
            intonation_scale=1.2,
        ),
    )
    np.testing.assert_array_equal(rs_result, wave + 3.0)
    assert calls == ["td-psola", "resampling"]


def test_process_wave_rejects_projected_oversized_silence_or_resampling(
    v2_fake_audio_manager_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = v2_fake_audio_manager_cls()
    wave = np.ones(16, dtype=np.float32)
    monkeypatch.setattr(audio_helpers, "MAX_GENERATED_WAVE_SAMPLES", 32)

    with pytest.raises(audio_helpers.AudioValidationError, match="maximum of 32"):
        process_wave(manager, wave, 16000, _options(pre_phoneme_length=0.01))

    with pytest.raises(audio_helpers.AudioValidationError, match="maximum of 32"):
        process_wave(manager, wave, 16000, _options(output_sampling_rate=48000))


def test_process_wave_applies_adjusted_f0_before_pause_replacement(
    v2_fake_audio_manager_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = v2_fake_audio_manager_cls()
    events: list[str] = []
    wave = np.linspace(-0.25, 0.25, 8, dtype=np.float32)

    def fake_world(
        audio_manager: object,
        current: np.ndarray,
        sampling_rate: int,
        pitch_scale: float,
        intonation_scale: float,
        adjusted_f0: np.ndarray,
    ) -> np.ndarray:
        assert audio_manager is manager
        assert sampling_rate == 16000
        assert pitch_scale == 0.0
        assert intonation_scale == 1.0
        np.testing.assert_array_equal(adjusted_f0, [110.0, 120.0])
        events.append("adjusted_f0")
        return current + 1.0

    def fake_pause(
        current: np.ndarray, *_args: object, **_kwargs: object
    ) -> np.ndarray:
        np.testing.assert_array_equal(current, wave + 1.0)
        events.append("pause")
        return current

    monkeypatch.setattr(wave_processing, "_world_process", fake_world)
    monkeypatch.setattr(audio_helpers, "replace_pause_segments", fake_pause)

    result, sampling_rate = process_wave(
        manager,
        wave,
        16000,
        _options(
            processing_algorithm="world",
            adjusted_f0=[110.0, 120.0],
            pause_length=0.3,
            pause_start_trim_buffer=0.0,
            pause_end_trim_buffer=0.0,
            mora_durations=_internal_pause_durations(),
        ),
    )

    assert sampling_rate == 16000
    np.testing.assert_array_equal(result, wave + 1.0)
    assert events == ["adjusted_f0", "pause"]


def test_world_processing_preserves_unvoiced_f0_frames_and_fits_track_length(
    v2_fake_audio_manager_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = v2_fake_audio_manager_cls()
    wave = np.zeros(32, dtype=np.float32)
    captured: dict[str, np.ndarray] = {}

    def fake_world(
        _wave: np.ndarray, _sampling_rate: int
    ) -> tuple[np.ndarray, None, None]:
        return np.array([0.0, 100.0, 200.0, 0.0]), None, None

    def fake_synthesize(
        f0: np.ndarray,
        _spectral_envelope: object,
        _aperiodicity: object,
        _sampling_rate: int,
    ) -> np.ndarray:
        captured["f0"] = np.asarray(f0).copy()
        return np.zeros(32, dtype=np.float64)

    manager.get_world = fake_world
    monkeypatch.setattr(load_pyworld(), "synthesize", fake_synthesize)

    result = wave_processing._world_process(
        manager,
        wave,
        16000,
        pitch_scale=0.0,
        intonation_scale=0.5,
        adjusted_f0=[0.0, 100.0, 200.0, 0.0],
    )

    np.testing.assert_array_equal(captured["f0"], [0.0, 125.0, 175.0, 0.0])
    assert result.dtype == np.float32
    assert result.shape == wave.shape

    # 長さが異なるF0軌跡も基準フレーム数へ線形補間して渡される。
    wave_processing._world_process(
        manager,
        wave,
        16000,
        pitch_scale=0.0,
        intonation_scale=1.0,
        adjusted_f0=[100.0, 400.0],
    )
    np.testing.assert_allclose(captured["f0"], [100.0, 200.0, 300.0, 400.0])
