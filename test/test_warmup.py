import numpy as np
import pytest

import run as engine_run

SPEAKER_A = "00000000-0000-0000-0000-00000000000a"
SPEAKER_B = "00000000-0000-0000-0000-00000000000b"


class FakeMetaManager:
    def __init__(self, metas: list[dict]):
        self._metas = metas

    def get_metas_dict(self) -> list[dict]:
        return self._metas


class FakeAudioManager:
    """実モデルを読み込まず、ウォームアップの呼び出しだけを記録する。"""

    def __init__(
        self,
        metas: list[dict],
        initialized: set[tuple[str, int]] | None = None,
        error: Exception | None = None,
        failing_style_ids: set[int] | None = None,
    ):
        self.meta_manager = FakeMetaManager(metas)
        self.initialized = initialized or set()
        self.error = error
        self.failing_style_ids = failing_style_ids
        self.synthesis_calls: list[dict] = []

    def is_speaker_initialized(self, style_id: int, speaker_uuid: str | None = None):
        return (speaker_uuid, style_id) in self.initialized

    def synthesis(self, text, style_id, **kwargs):
        self.synthesis_calls.append({"text": text, "style_id": style_id, **kwargs})
        if self.error is not None and (
            self.failing_style_ids is None or style_id in self.failing_style_ids
        ):
            raise self.error
        return np.zeros(8, dtype=np.float32)


def _metas() -> list[dict]:
    return [
        {
            "name": "A",
            "speaker_uuid": SPEAKER_A,
            "styles": [{"name": "a0", "id": 10}, {"name": "a1", "id": 11}],
        },
        {
            "name": "B",
            "speaker_uuid": SPEAKER_B,
            "styles": [{"name": "b0", "id": 20}],
        },
    ]


def test_warm_up_warms_all_loaded_models_in_public_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A/a1・B/b0だけが読み込み済み。未ロードのA/a0は温めない。
    manager = FakeAudioManager(_metas(), initialized={(SPEAKER_B, 20), (SPEAKER_A, 11)})

    engine_run._warm_up(manager)

    assert manager.synthesis_calls == [
        {
            "text": "こんにちは",
            "style_id": 11,
            "speaker_uuid": SPEAKER_A,
            "output_sampling_rate": 24000,
        },
        {
            "text": "こんにちは",
            "style_id": 20,
            "speaker_uuid": SPEAKER_B,
            "output_sampling_rate": 24000,
        },
    ]
    error_output = capsys.readouterr().err
    assert "INFO:" in error_output
    assert "2/2" in error_output


def test_warm_up_skips_when_no_model_is_loaded(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = FakeAudioManager(_metas())

    engine_run._warm_up(manager)

    assert manager.synthesis_calls == []
    error_output = capsys.readouterr().err
    assert "WARNING:" in error_output
    assert "--max-loaded-models" in error_output


def test_warm_up_skips_without_styles(capsys: pytest.CaptureFixture[str]) -> None:
    manager = FakeAudioManager([])

    engine_run._warm_up(manager)

    assert manager.synthesis_calls == []
    assert "WARNING:" in capsys.readouterr().err


def test_warm_up_failure_does_not_stop_other_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = FakeAudioManager(
        _metas(),
        initialized={(SPEAKER_A, 10), (SPEAKER_B, 20)},
        error=RuntimeError("boom"),
        failing_style_ids={10},
    )

    engine_run._warm_up(manager)

    assert [call["style_id"] for call in manager.synthesis_calls] == [10, 20]
    error_output = capsys.readouterr().err
    assert "WARNING:" in error_output
    assert "boom" in error_output
    assert "1/2" in error_output
