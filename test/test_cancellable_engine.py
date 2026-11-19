"""キャンセル可能合成ワーカーのプロセス管理・切断検知・エラー伝播・終了処理を検証する。"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from tempfile import NamedTemporaryFile as RealNamedTemporaryFile
from threading import RLock
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from coeirocore.coeiro_manager import (
    InvalidSynthesisParameterError,
    StyleNotFoundError,
)
from fastapi import FastAPI, HTTPException, Request

import run as engine_run
from run import _add_cors_middleware, _add_exception_handlers, _create_lifespan
from voicevox_engine import cancellable_engine as cancellable_module
from voicevox_engine.cancellable_engine import (
    CancellableEngine,
    CancellableRequestDisconnected,
    CancellableWorkerError,
    CancellableWorkerSynthesisError,
    _receive_worker_response,
    _RequestState,
    start_synthesis_subprocess,
)
from voicevox_engine.setting import CorsPolicyMode


class _FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.join_timeouts: list[float | None] = []
        self.join_thread_ids: list[int] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        self.join_thread_ids.append(threading.get_ident())

    def close(self) -> None:
        self.close_calls += 1


def _worker_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "device": "cpu",
        "use_gpu": None,
        "device_index": 0,
        "opencl_platform_index": 0,
        "resampler": "resampy",
        "voicelib_dir": [],
        "voicevox_dir": None,
        "runtime_dir": [],
        "cpu_num_threads": 0,
        "speaker_info_dir": None,
        "enable_mock": True,
        "voice_smoothing": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _engine_without_processes() -> CancellableEngine:
    engine = object.__new__(CancellableEngine)
    engine._state_lock = RLock()
    engine._shutting_down = False
    engine._disconnection_poll_interval = 0.01
    engine.watch_con_list = []
    engine.procs_and_cons = queue.Queue()
    engine.start_new_proc = Mock()
    engine._mp_context = get_context("spawn")
    return engine


def test_cancellable_engine_requires_at_least_one_worker() -> None:
    args = argparse.Namespace(enable_cancellable_synthesis=True, init_processes=0)

    with pytest.raises(ValueError, match="init_processes must be at least 1"):
        CancellableEngine(args)


def test_eof_replaces_failed_worker() -> None:
    engine = _engine_without_processes()
    request = Mock()
    process = Mock()
    process.is_alive.return_value = True
    connection = Mock()
    connection.recv.side_effect = EOFError
    replacement = (Mock(), Mock())
    engine.start_new_proc.return_value = replacement
    engine.procs_and_cons.put((process, connection))

    with pytest.raises(CancellableWorkerError) as error:
        engine._synthesis_impl(
            query=Mock(),
            speaker_id=1,
            request=request,
            enable_interrogative_upspeak=True,
            core_version=None,
        )

    assert not isinstance(error.value, HTTPException)
    assert "通信に失敗" in str(error.value)
    assert engine.procs_and_cons.get_nowait() == replacement
    engine.start_new_proc.assert_called_once_with()
    connection.close.assert_called_once_with()
    assert engine.watch_con_list == []


def test_dead_worker_is_not_requeued_when_finalized() -> None:
    engine = _engine_without_processes()
    request = Mock()
    process = _FakeProcess(alive=False)
    connection = Mock()
    replacement = (_FakeProcess(), Mock())
    engine.start_new_proc.return_value = replacement
    engine.watch_con_list.append(_RequestState(request, process, connection))

    engine.finalize_con(request, process, connection)

    assert engine.procs_and_cons.get_nowait() == replacement
    assert engine.procs_and_cons.empty()
    assert process.close_calls == 1


def test_concurrent_finalizers_replace_a_worker_once() -> None:
    engine = _engine_without_processes()
    request = Mock()
    process = _FakeProcess()
    connection = Mock()
    engine.watch_con_list.append(_RequestState(request, process, connection))
    barrier = threading.Barrier(2)

    def finalize() -> None:
        barrier.wait()
        engine.finalize_con(request, process, connection)

    threads = [threading.Thread(target=finalize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert engine.procs_and_cons.qsize() == 1
    engine.start_new_proc.assert_not_called()


def test_child_error_is_structured_and_not_treated_as_http_422() -> None:
    connection = Mock()
    connection.recv.return_value = {
        "kind": "error",
        "exception_type": "ValueError",
        "message": "bad parameter",
        "traceback": "Traceback (most recent call last): ...",
    }

    with pytest.raises(CancellableWorkerSynthesisError) as error:
        _receive_worker_response(connection)

    assert error.value.exception_type == "ValueError"
    assert error.value.message == "bad parameter"
    assert error.value.child_traceback.startswith("Traceback")
    assert "Traceback (most recent call last)" in str(error.value)
    assert not isinstance(error.value, HTTPException)


def test_child_reports_synthesis_error_and_keeps_worker_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def __init__(self, *requests: object) -> None:
            self.requests = iter(requests)
            self.sent: list[dict[str, object]] = []
            self.closed = False

        def recv(self) -> object:
            return next(self.requests)

        def send(self, message: dict[str, object]) -> None:
            self.sent.append(message)

        def close(self) -> None:
            self.closed = True

    query = Mock(outputSamplingRate=16_000)
    synthesis_engine = Mock()
    synthesis_engine.synthesis.side_effect = ValueError("invalid test input")
    connection = FakeConnection((query, 1, True, None), {"kind": "shutdown"})
    monkeypatch.setattr(cancellable_module, "resolve_device", lambda *_a, **_kw: "cpu")
    monkeypatch.setattr(
        cancellable_module,
        "make_synthesis_engines",
        lambda *_a, **_kw: {"1.0": synthesis_engine},
    )

    start_synthesis_subprocess(_worker_args(generator_only=False), connection)

    assert connection.closed
    assert connection.sent[0]["kind"] == "error"
    assert str(connection.sent[0]["exception_type"]).endswith("ValueError")
    assert len(connection.sent) == 1
    synthesis_engine.synthesis.assert_called_once()


def test_subprocess_removes_wave_if_result_cannot_be_sent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingConnection:
        def __init__(self) -> None:
            self.received = False

        def recv(self) -> tuple[Mock, int, bool, None]:
            if self.received:
                raise EOFError
            self.received = True
            return Mock(outputSamplingRate=16_000), 1, False, None

        def send(self, _message: object) -> None:
            raise OSError("connection closed")

        def close(self) -> None:
            pass

    synthesis_engine = Mock()
    synthesis_engine.synthesis.return_value = np.zeros(16, dtype=np.float32)
    monkeypatch.setattr(cancellable_module, "resolve_device", lambda *_a, **_kw: "cpu")
    monkeypatch.setattr(
        cancellable_module,
        "make_synthesis_engines",
        lambda *_a, **_kw: {"1.0": synthesis_engine},
    )
    monkeypatch.setattr(
        cancellable_module,
        "NamedTemporaryFile",
        partial(RealNamedTemporaryFile, dir=tmp_path),
    )

    start_synthesis_subprocess(_worker_args(generator_only=False), FailingConnection())

    assert list(tmp_path.iterdir()) == []


def test_disconnection_monitor_cancels_request_waiting_for_worker() -> None:
    engine = _engine_without_processes()
    request = Mock()
    request.is_disconnected = AsyncMock(return_value=True)

    async def run_test() -> None:
        monitor = asyncio.create_task(engine.catch_disconnection())
        try:
            with pytest.raises(CancellableRequestDisconnected):
                await asyncio.wait_for(
                    asyncio.to_thread(
                        engine._synthesis_impl,
                        Mock(),
                        1,
                        request,
                        True,
                        None,
                    ),
                    timeout=1,
                )
            request.is_disconnected.assert_awaited()
        finally:
            engine.shutdown(timeout=0)
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    asyncio.run(run_test())


def test_disconnect_reaches_monitor_through_origin_middleware(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    engine = _engine_without_processes()
    app = FastAPI(lifespan=_create_lifespan(engine, Mock()))
    _add_exception_handlers(app)
    _add_cors_middleware(app, CorsPolicyMode.localapps, None)
    monkeypatch.setattr(engine_run, "update_dict", lambda: None)

    async def run_test() -> None:
        started = asyncio.Event()
        incoming: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        incoming.put_nowait({"type": "http.request", "body": b""})
        sent: list[dict[str, object]] = []

        @app.post("/synthesis")
        async def synthesis(request: Request) -> object:
            await request.body()
            started.set()
            return await asyncio.to_thread(
                engine._synthesis_impl, Mock(), 1, request, False, None
            )

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "method": "POST",
            "scheme": "http",
            "path": "/synthesis",
            "query_string": b"",
            "headers": [(b"origin", b"http://localhost")],
        }
        async with app.router.lifespan_context(app):
            task = asyncio.create_task(app(scope, incoming.get, send))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                incoming.put_nowait({"type": "http.disconnect"})
                await asyncio.wait_for(asyncio.shield(task), timeout=1)
            finally:
                engine.shutdown(timeout=0)
                await asyncio.gather(task, return_exceptions=True)

        assert not engine.watch_con_list
        assert sent[0]["status"] == 499

    asyncio.run(run_test())
    assert not [record for record in caplog.records if record.exc_info]


def test_async_disconnection_cleanup_does_not_join_on_event_loop_thread() -> None:
    engine = _engine_without_processes()
    request = Mock()
    process = _FakeProcess()
    connection = Mock()
    replacement = (_FakeProcess(), Mock())
    engine.start_new_proc.return_value = replacement
    state = _RequestState(request, process, connection)
    engine.watch_con_list.append(state)
    event_loop_thread_id = threading.get_ident()

    async def run_test() -> None:
        await engine._finalize_disconnected(state)

    asyncio.run(run_test())

    assert process.join_thread_ids
    assert all(
        thread_id != event_loop_thread_id for thread_id in process.join_thread_ids
    )
    engine.shutdown(timeout=0)


def test_shutdown_is_explicit_bounded_and_idempotent() -> None:
    engine = _engine_without_processes()
    process = _FakeProcess()
    connection = Mock()
    connection.send.side_effect = lambda _message: setattr(process, "alive", False)
    engine.procs_and_cons.put((process, connection))

    engine.shutdown(timeout=0.5)
    engine.shutdown(timeout=0.5)

    connection.send.assert_called_once_with({"kind": "shutdown"})
    assert process.join_timeouts
    assert process.close_calls == 1
    assert engine.procs_and_cons.empty()


def test_shutdown_collects_a_worker_being_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_without_processes()
    process = _FakeProcess()
    connection = Mock()
    checking = threading.Event()
    resume = threading.Event()
    stopped = threading.Event()

    def is_alive() -> bool:
        checking.set()
        assert resume.wait(timeout=2)
        return process.alive

    def shutdown() -> None:
        engine.shutdown(timeout=0)
        stopped.set()

    monkeypatch.setattr(process, "is_alive", is_alive)
    with ThreadPoolExecutor(max_workers=2) as executor:
        returning = executor.submit(engine._put_available_worker, (process, connection))
        try:
            assert checking.wait(timeout=2)
            stopping = executor.submit(shutdown)
            # 旧実装ではこの間にshutdownが完了し、後からキューへ戻るワーカーが取り残された。
            stopped.wait(timeout=0.05)
        finally:
            resume.set()
        returning.result(timeout=2)
        stopping.result(timeout=2)

    assert engine.procs_and_cons.empty()
    assert process.close_calls == 1


def test_workers_use_spawn_context(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(enable_cancellable_synthesis=True, init_processes=1)
    monkeypatch.setattr(
        CancellableEngine, "start_new_proc", lambda _self: (Mock(), Mock())
    )

    engine = CancellableEngine(args)

    assert engine._mp_context.get_start_method() == "spawn"


def test_subprocess_passes_device_without_preloading_every_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _worker_args(
        device="opencl",
        device_index=2,
        opencl_platform_index=1,
        speaker_info_dir=Path("speaker_info"),
        resampler="soxr-vhq",
        max_loaded_models=None,
        voice_smoothing=True,
        dangerous_skip_check=True,
    )
    make_engines = Mock(side_effect=RuntimeError("stop before worker loop"))
    monkeypatch.setattr(cancellable_module, "make_synthesis_engines", make_engines)

    with pytest.raises(RuntimeError, match="stop before worker loop"):
        start_synthesis_subprocess(args, Mock())

    make_engines.assert_called_once_with(
        device="opencl",
        device_index=2,
        opencl_platform_index=1,
        resampler="soxr-vhq",
        voicelib_dirs=[],
        voicevox_dir=None,
        runtime_dirs=[],
        cpu_num_threads=0,
        speaker_info_dir=Path("speaker_info"),
        enable_mock=True,
        max_loaded_models=1,
        generator_only=True,
        voice_smoothing=True,
        dangerous_skip_check=True,
    )


def test_disabled_cancellable_synthesis_returns_explicit_404(
    engine_client_factory,
) -> None:
    client, _ = engine_client_factory.create_client()
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)

    response = client.post(
        "/voicevox/cancellable_synthesis",
        params={"speaker": style_id},
        json=query,
    )

    assert response.status_code == 404
    assert "デフォルトで無効" in response.json()["detail"]


def test_enabled_cancellable_synthesis_uses_injected_engine(
    engine_client_factory,
) -> None:
    wav_path = engine_client_factory.tmp_path / "cancellable.wav"
    wav_path.write_bytes(b"RIFF-cancellable-test")
    cancellable_engine = Mock()
    cancellable_engine._synthesis_impl.return_value = str(wav_path)
    client, _ = engine_client_factory.create_client(
        cancellable_engine=cancellable_engine
    )
    style_id = engine_client_factory.style_id
    query = engine_client_factory.post_audio_query(client)

    response = client.post(
        "/voicevox/cancellable_synthesis",
        params={
            "speaker": style_id,
            "enable_interrogative_upspeak": False,
        },
        json=query,
    )

    assert response.status_code == 200
    assert response.content == b"RIFF-cancellable-test"
    assert (
        cancellable_engine._synthesis_impl.call_args.kwargs[
            "enable_interrogative_upspeak"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (InvalidSynthesisParameterError("speed_scale must be positive"), 422),
        (StyleNotFoundError("style was removed"), 404),
        (ValueError("unexpected implementation error"), 500),
    ],
)
def test_cancellable_worker_errors_keep_normal_synthesis_status(
    engine_client_factory,
    error: Exception,
    expected_status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cancellable_engine = Mock()
    cancellable_engine._synthesis_impl.side_effect = CancellableWorkerSynthesisError(
        {
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": "worker traceback",
        }
    )
    client, _ = engine_client_factory.create_client(
        cancellable_engine=cancellable_engine
    )
    style_id = engine_client_factory.style_id

    response = client.post(
        "/voicevox/cancellable_synthesis",
        params={"speaker": style_id},
        json=engine_client_factory.post_audio_query(client),
    )

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(error)}
    if expected_status == 500:
        assert "worker traceback" in caplog.text


def test_cancellable_disconnection_monitor_follows_app_lifespan(
    engine_client_factory,
) -> None:
    class MonitoringCancellableEngine:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.shutdown = threading.Event()

        async def catch_disconnection(self) -> None:
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.stopped.set()

        async def shutdown_async(self) -> None:
            self.shutdown.set()

    cancellable_engine = MonitoringCancellableEngine()
    client, _ = engine_client_factory.create_client(
        cancellable_engine=cancellable_engine
    )

    with client:
        assert cancellable_engine.started.wait(timeout=1)

    assert cancellable_engine.stopped.wait(timeout=1)
    assert cancellable_engine.shutdown.wait(timeout=1)
