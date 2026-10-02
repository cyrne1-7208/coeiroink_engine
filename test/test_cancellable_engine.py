import argparse
import asyncio
import pickle
import queue
import threading
from multiprocessing import get_context
from pathlib import Path
from tempfile import NamedTemporaryFile as RealNamedTemporaryFile
from threading import RLock
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from voicevox_engine.cancellable_engine import (
    CancellableEngine,
    CancellableRequestDisconnected,
    CancellableWorkerError,
    CancellableWorkerSynthesisError,
    _receive_worker_response,
    _RequestState,
    start_synthesis_subprocess,
)


class _FakeProcess:
    def __init__(self, alive: bool = True):
        self.alive = alive
        self.join_timeouts = []
        self.join_thread_ids = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False

    def join(self, timeout):
        self.join_timeouts.append(timeout)
        self.join_thread_ids.append(threading.get_ident())

    def close(self):
        self.close_calls += 1


def _engine_without_processes():
    engine = object.__new__(CancellableEngine)
    engine._state_lock = RLock()
    engine._shutting_down = False
    engine._disconnection_poll_interval = 0.01
    engine.watch_con_list = []
    engine.procs_and_cons = queue.Queue()
    engine.start_new_proc = Mock()
    engine._mp_context = get_context("spawn")
    return engine


def test_cancellable_engine_requires_at_least_one_worker():
    args = argparse.Namespace(
        enable_cancellable_synthesis=True,
        init_processes=0,
    )

    with pytest.raises(ValueError, match="init_processes must be at least 1"):
        CancellableEngine(args)


def test_eof_replaces_failed_worker():
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


def test_worker_is_not_added_twice_when_finalizers_race():
    engine = _engine_without_processes()
    request = Mock()
    process = Mock()
    process.is_alive.return_value = True
    connection = Mock()
    engine.watch_con_list.append(_RequestState(request, process, connection))

    engine.finalize_con(request, process, connection)
    engine.finalize_con(request, process, None)

    assert engine.procs_and_cons.get_nowait() == (process, connection)
    assert engine.procs_and_cons.empty()
    engine.start_new_proc.assert_not_called()


def test_dead_worker_is_not_requeued_when_finalized():
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


def test_concurrent_finalizers_replace_a_worker_once():
    engine = _engine_without_processes()
    request = Mock()
    process = _FakeProcess()
    connection = Mock()
    engine.watch_con_list.append(_RequestState(request, process, connection))
    barrier = threading.Barrier(2)

    def finalize():
        barrier.wait()
        engine.finalize_con(request, process, connection)

    threads = [threading.Thread(target=finalize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert engine.procs_and_cons.qsize() == 1
    engine.start_new_proc.assert_not_called()


def test_child_error_is_structured_and_not_treated_as_http_422():
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


def test_child_reports_synthesis_error_and_keeps_worker_alive():
    class FakeConnection:
        def __init__(self, *requests):
            self.requests = iter(requests)
            self.sent = []
            self.closed = False

        def recv(self):
            return next(self.requests)

        def send(self, message):
            self.sent.append(message)

        def close(self):
            self.closed = True

    query = Mock(outputSamplingRate=16_000)
    synthesis_engine = Mock()
    synthesis_engine.synthesis.side_effect = ValueError("invalid test input")
    connection = FakeConnection(
        (query, 1, True, None),
        {"kind": "shutdown"},
    )
    args = argparse.Namespace(
        device="cpu",
        use_gpu=None,
        device_index=0,
        opencl_platform_index=0,
        resampler="resampy",
        voicelib_dir=[],
        voicevox_dir=None,
        runtime_dir=[],
        cpu_num_threads=0,
        speaker_info_dir=None,
        enable_mock=True,
        generator_only=False,
    )

    with (
        patch(
            "voicevox_engine.cancellable_engine.resolve_device",
            return_value="cpu",
        ),
        patch(
            "voicevox_engine.cancellable_engine.make_synthesis_engines",
            return_value={"1.0": synthesis_engine},
        ),
    ):
        start_synthesis_subprocess(args, connection)

    assert connection.closed
    assert connection.sent[0]["kind"] == "error"
    assert connection.sent[0]["exception_type"].endswith("ValueError")
    assert len(connection.sent) == 1
    synthesis_engine.synthesis.assert_called_once()


def test_subprocess_removes_wave_if_result_cannot_be_sent(tmp_path: Path):
    class FailingConnection:
        def __init__(self):
            self.received = False

        def recv(self):
            if self.received:
                raise EOFError
            self.received = True
            return Mock(outputSamplingRate=16_000), 1, False, None

        def send(self, _message):
            raise OSError("connection closed")

        def close(self):
            pass

    synthesis_engine = Mock()
    synthesis_engine.synthesis.return_value = np.zeros(16, dtype=np.float32)
    args = argparse.Namespace(
        device="cpu",
        use_gpu=None,
        device_index=0,
        opencl_platform_index=0,
        resampler="resampy",
        voicelib_dir=[],
        voicevox_dir=None,
        runtime_dir=[],
        cpu_num_threads=0,
        speaker_info_dir=None,
        enable_mock=True,
        generator_only=False,
    )

    def temporary_file(**kwargs):
        return RealNamedTemporaryFile(dir=tmp_path, **kwargs)

    with (
        patch(
            "voicevox_engine.cancellable_engine.resolve_device",
            return_value="cpu",
        ),
        patch(
            "voicevox_engine.cancellable_engine.make_synthesis_engines",
            return_value={"1.0": synthesis_engine},
        ),
        patch(
            "voicevox_engine.cancellable_engine.NamedTemporaryFile",
            side_effect=temporary_file,
        ),
    ):
        start_synthesis_subprocess(args, FailingConnection())

    assert list(tmp_path.iterdir()) == []


def test_disconnection_monitor_cancels_request_waiting_for_worker():
    engine = _engine_without_processes()
    request = Mock()
    request.is_disconnected = AsyncMock(return_value=True)

    async def run_test():
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


def test_async_disconnection_cleanup_does_not_join_on_event_loop_thread():
    engine = _engine_without_processes()
    request = Mock()
    process = _FakeProcess()
    connection = Mock()
    replacement = (_FakeProcess(), Mock())
    engine.start_new_proc.return_value = replacement
    state = _RequestState(request, process, connection)
    engine.watch_con_list.append(state)
    event_loop_thread_id = threading.get_ident()

    async def run_test():
        await engine._finalize_disconnected(state)

    asyncio.run(run_test())

    assert process.join_thread_ids
    assert all(
        thread_id != event_loop_thread_id for thread_id in process.join_thread_ids
    )
    engine.shutdown(timeout=0)


def test_shutdown_is_explicit_bounded_and_idempotent():
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


def test_workers_use_spawn_context():
    args = argparse.Namespace(
        enable_cancellable_synthesis=True,
        init_processes=1,
    )
    with patch.object(
        CancellableEngine,
        "start_new_proc",
        return_value=(Mock(), Mock()),
    ):
        engine = CancellableEngine(args)

    assert engine._mp_context.get_start_method() == "spawn"


def test_subprocess_passes_device_without_preloading_every_model():
    args = argparse.Namespace(
        device="opencl",
        use_gpu=None,
        device_index=2,
        opencl_platform_index=1,
        voicelib_dir=[],
        voicevox_dir=None,
        runtime_dir=[],
        cpu_num_threads=0,
        speaker_info_dir=Path("speaker_info"),
        enable_mock=True,
        resampler="soxr-vhq",
        max_loaded_models=None,
        generator_only=True,
    )

    with (
        patch(
            "voicevox_engine.cancellable_engine.make_synthesis_engines",
            side_effect=RuntimeError("stop before worker loop"),
        ) as make_engines,
        pytest.raises(RuntimeError, match="stop before worker loop"),
    ):
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
    )


def test_subprocess_target_and_arguments_are_spawn_picklable():
    """Windowsのspawn方式でワーカー生成前にpickleエラーを起こさないことを保証する。"""

    args = argparse.Namespace(
        device="directml",
        use_gpu=None,
        device_index=1,
        opencl_platform_index=0,
        voicelib_dir=[Path("voice")],
        voicevox_dir=Path("engine"),
        runtime_dir=[Path("runtime")],
        cpu_num_threads=2,
        speaker_info_dir=Path("speaker_info"),
        enable_mock=True,
    )

    assert pickle.loads(pickle.dumps(start_synthesis_subprocess)) is (
        start_synthesis_subprocess
    )
    restored = pickle.loads(pickle.dumps(args))
    assert vars(restored) == vars(args)
