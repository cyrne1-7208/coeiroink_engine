import argparse
import pickle
import queue
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from voicevox_engine.cancellable_engine import (
    CancellableEngine,
    start_synthesis_subprocess,
)


def _engine_without_processes():
    engine = object.__new__(CancellableEngine)
    engine.watch_con_list = []
    engine.procs_and_cons = queue.Queue()
    engine.start_new_proc = Mock()
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
    process.is_alive.return_value = False
    connection = Mock()
    connection.recv.side_effect = EOFError
    replacement = (Mock(), Mock())
    engine.start_new_proc.return_value = replacement
    engine.procs_and_cons.put((process, connection))

    with pytest.raises(HTTPException) as error:
        engine._synthesis_impl(
            query=Mock(),
            speaker_id=1,
            request=request,
            enable_interrogative_upspeak=True,
            core_version=None,
        )

    assert error.value.status_code == 422
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
    engine.watch_con_list.append((request, process))

    engine.finalize_con(request, process, connection)
    engine.finalize_con(request, process, None)

    assert engine.procs_and_cons.get_nowait() == (process, connection)
    assert engine.procs_and_cons.empty()
    engine.start_new_proc.assert_not_called()


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
        load_all_models=True,
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
        load_all_models=False,
    )


def test_subprocess_target_and_arguments_are_spawn_picklable():
    """Windowsのspawn方式でワーカ生成前にpickleエラーを起こさないことを保証する。"""

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
