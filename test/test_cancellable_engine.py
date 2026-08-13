import queue
from unittest.mock import Mock

from fastapi import HTTPException
import pytest

from voicevox_engine.cancellable_engine import CancellableEngine


def _engine_without_processes():
    engine = object.__new__(CancellableEngine)
    engine.watch_con_list = []
    engine.procs_and_cons = queue.Queue()
    engine.start_new_proc = Mock()
    return engine


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
