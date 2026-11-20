# 起動時ウォームアップと、辞書・モデル準備・ワーカー終了を行うlifespanの順序のテスト

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from coeirocore.coeiro_manager import ModelLoadError
from fastapi import FastAPI

import run as engine_run

SPEAKER_A = "00000000-0000-0000-0000-00000000000a"
SPEAKER_B = "00000000-0000-0000-0000-00000000000b"


class FakeMetaManager:
    def __init__(self, metas: list[dict]) -> None:
        self._metas = metas

    def get_metas_dict(self) -> list[dict]:
        return self._metas


# 実モデルを読み込まず、ウォームアップの呼び出しだけを記録するフェイク
class FakeAudioManager:
    def __init__(
        self,
        metas: list[dict],
        initialized: set[tuple[str, int]] | None = None,
        error: Exception | None = None,
        failing_style_ids: set[int] | None = None,
    ) -> None:
        self.meta_manager = FakeMetaManager(metas)
        self.initialized = initialized or set()
        self.error = error
        self.failing_style_ids = failing_style_ids
        self.synthesis_calls: list[dict] = []

    def is_speaker_initialized(
        self, style_id: int, speaker_uuid: str | None = None
    ) -> bool:
        return (speaker_uuid, style_id) in self.initialized

    def synthesis(self, text: list[str], style_id: int, **kwargs: object) -> np.ndarray:
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

    tokens = engine_run.estimate_prosody("こんにちは").plain
    assert manager.synthesis_calls == [
        {
            "text": tokens,
            "style_id": 11,
            "speaker_uuid": SPEAKER_A,
            "output_sampling_rate": 24000,
        },
        {
            "text": tokens,
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


def test_warm_up_analysis_failure_does_not_stop_startup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = FakeAudioManager(_metas(), initialized={(SPEAKER_A, 10)})

    def fail_analysis(_text: str) -> None:
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(engine_run, "estimate_prosody", fail_analysis)

    engine_run._warm_up(manager)

    assert manager.synthesis_calls == []
    assert "analysis failed" in capsys.readouterr().err


def test_startup_prepares_models_after_dictionary_and_before_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager = SimpleNamespace(prepare_models=lambda: events.append("prepare"))
    monkeypatch.setattr(engine_run, "update_dict", lambda: events.append("dictionary"))
    monkeypatch.setattr(engine_run, "_warm_up", lambda _: events.append("warmup"))
    app = FastAPI(lifespan=engine_run._create_lifespan(None, manager, warmup=True))

    async def start() -> None:
        async with app.router.lifespan_context(app):
            events.append("ready")

    asyncio.run(start())

    assert events == ["dictionary", "prepare", "warmup", "ready"]


def test_startup_memory_failure_stops_startup_and_shuts_down_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = SimpleNamespace(
        prepare_models=Mock(side_effect=ModelLoadError("insufficient memory"))
    )
    workers = SimpleNamespace(
        catch_disconnection=AsyncMock(), shutdown_async=AsyncMock()
    )
    monkeypatch.setattr(engine_run, "update_dict", lambda: None)
    warm_up = Mock()
    monkeypatch.setattr(engine_run, "_warm_up", warm_up)
    app = FastAPI(lifespan=engine_run._create_lifespan(workers, manager, warmup=True))

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("startup must not complete after a model preparation error")

    with pytest.raises(ModelLoadError, match="insufficient memory"):
        asyncio.run(start())

    warm_up.assert_not_called()
    workers.shutdown_async.assert_awaited_once_with()
