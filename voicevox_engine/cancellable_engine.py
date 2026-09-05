import argparse
import asyncio
import queue
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing import Process, get_context
from multiprocessing.connection import Connection
from tempfile import NamedTemporaryFile
from threading import Event, RLock
from typing import Any

import soundfile

# FIXME: ワーカー層からFastAPIへの依存を取り除く。
from fastapi import HTTPException, Request
from packaging.version import Version

from .model import AudioQuery
from .synthesis_engine import make_synthesis_engines
from .synthesis_engine.make_synthesis_engines import resolve_device
from .utility import delete_file

_WORKER_WAIT_TIMEOUT = 0.1
_DISCONNECTION_POLL_INTERVAL = 1.0
_PROCESS_JOIN_TIMEOUT = 1.0


@dataclass
class _RequestState:
    """待機中または合成中のリクエストとワーカーの所有関係を保持する。"""

    request: Request
    proc: Process | None = None
    connection: Connection | None = None
    cancelled: Event = field(default_factory=Event)
    finalized: bool = False


class CancellableWorkerError(RuntimeError):
    """キャンセル用ワーカーとの通信または終了に関するエラー。"""

    reusable = False


class CancellableWorkerSynthesisError(CancellableWorkerError):
    """子プロセスの合成例外を構造化したまま親へ伝えるエラー。"""

    reusable = True

    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        self.exception_type = payload.get("exception_type", "Exception")
        self.message = payload.get("message", "")
        self.child_traceback = payload.get("traceback", "")
        details = (
            "サブプロセスの音声合成に失敗しました: "
            f"{self.exception_type}: {self.message}"
        )
        if self.child_traceback:
            details = f"{details}\n{self.child_traceback.rstrip()}"
        super().__init__(details)


class CancellableRequestDisconnected(CancellableWorkerError):
    """ワーカー取得前または合成中にHTTP接続が切断された場合のエラー。"""


def _child_error_payload(error: Exception) -> dict[str, str]:
    """pickleできない例外本体の代わりに、診断に必要な情報だけを送る。"""

    return {
        "kind": "error",
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _is_process_alive(proc: Process) -> bool:
    try:
        return proc.is_alive()
    except (AssertionError, ValueError):
        return False


def _close_connection(connection: Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except (OSError, ValueError):
        pass


def _join_process(proc: Process, timeout: float) -> None:
    try:
        proc.join(max(0.0, timeout))
    except (AssertionError, OSError, ValueError):
        pass


def _close_process(proc: Process) -> None:
    try:
        proc.close()
    except (AssertionError, OSError, ValueError):
        pass


def _stop_process(
    proc: Process,
    connection: Connection | None,
    timeout: float,
    *,
    graceful: bool = False,
) -> None:
    """プロセスを期限内に停止し、PipeとProcessを閉じる。"""

    deadline = time.monotonic() + max(0.0, timeout)
    if graceful and connection is not None:
        try:
            connection.send({"kind": "shutdown"})
        except (EOFError, OSError, ValueError):
            graceful = False
    if not graceful:
        _close_connection(connection)
        connection = None
        try:
            if _is_process_alive(proc):
                proc.terminate()
        except (OSError, ValueError):
            pass

    _join_process(proc, max(0.0, deadline - time.monotonic()))
    if _is_process_alive(proc):
        try:
            proc.terminate()
        except (OSError, ValueError):
            pass
        _join_process(proc, max(0.0, deadline - time.monotonic()))
    if _is_process_alive(proc):
        kill = getattr(proc, "kill", None)
        if kill is not None:
            try:
                kill()
            except (OSError, ValueError):
                pass
        _join_process(proc, max(0.0, deadline - time.monotonic()))
    _close_connection(connection)
    _close_process(proc)


def _receive_worker_response(connection: Connection) -> str:
    response: Any = connection.recv()
    # 旧ワーカーとの一時的な互換性を保ち、文字列は成功応答として受け取る。
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        raise CancellableWorkerError("サブプロセスから不正な応答を受信しました")
    kind = response.get("kind")
    if kind == "result" and isinstance(response.get("path"), str):
        return response["path"]
    if kind == "error":
        raise CancellableWorkerSynthesisError(
            {key: str(value) for key, value in response.items() if key != "kind"}
        )
    raise CancellableWorkerError("サブプロセスから不正な応答種別を受信しました")


def _version_key(version: str) -> Version:
    """`+cpu`などのローカル接尾辞を含むCoreバージョンを比較可能な値へ変換する。"""
    return Version(version)


class CancellableEngine:
    """
    マルチプロセスで音声合成を実行し、クライアント切断時に処理をキャンセルできるエンジン。

    Attributes
    ----------
    watch_con_list: List[_RequestState]
        待機中を含むリクエストの状態を保持する。状態の確定はロック下で一度だけ行う
    procs_and_cons: queue.Queue[Tuple[Process, Connection]]
        音声合成の準備が完了したプロセスとコネクションを保持するQueue
        （音声合成中のプロセスは入っていない）
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """指定された数の音声合成ワーカープロセスを事前起動し、プールを初期化する。"""
        self.args = args
        if not self.args.enable_cancellable_synthesis:
            raise HTTPException(
                status_code=404,
                detail="実験的機能はデフォルトで無効になっています。使用するには引数を指定してください。",
            )
        if self.args.init_processes < 1:
            raise ValueError("init_processes must be at least 1")

        self._state_lock = RLock()
        self._shutting_down = False
        self._disconnection_poll_interval = _DISCONNECTION_POLL_INTERVAL
        self.watch_con_list: list[_RequestState] = []
        self.procs_and_cons: queue.Queue[tuple[Process, Connection]] = queue.Queue()
        # CUDAなどのランタイムをfork後に複製しないため、キャンセル用ワーカーは常にspawnで起動する。
        self._mp_context = get_context("spawn")
        for _ in range(self.args.init_processes):
            self.procs_and_cons.put(self.start_new_proc())

    def start_new_proc(
        self,
    ) -> tuple[Process, Connection]:
        """
        新しく開始したプロセスを返す関数

        Returns
        -------
        ret_proc: Process
            新規のプロセス
        sub_proc_con1: Connection
            ret_procのプロセスと通信するためのPipe
        """
        sub_proc_con1, sub_proc_con2 = self._mp_context.Pipe(True)
        # targetにはモジュール直下の関数だけを渡し、spawnで引数をpickleできるようにする。
        ret_proc = self._mp_context.Process(
            target=start_synthesis_subprocess,
            kwargs={
                "args": self.args,
                "sub_proc_con": sub_proc_con2,
            },
            daemon=True,
        )
        try:
            ret_proc.start()
        finally:
            _close_connection(sub_proc_con2)
        return ret_proc, sub_proc_con1

    def _register_request(self, request: Request) -> _RequestState:
        state = _RequestState(request=request)
        with self._state_lock:
            if self._shutting_down:
                raise CancellableWorkerError("キャンセル用ワーカーは終了処理中です")
            self.watch_con_list.append(state)
        return state

    def _claim_state(self, state: _RequestState) -> bool:
        """リクエストの後処理を一つの経路だけが担当するように確定する。"""

        with self._state_lock:
            if state.finalized:
                return False
            state.finalized = True
            for index, current in enumerate(self.watch_con_list):
                if current is state:
                    del self.watch_con_list[index]
                    break
        return True

    def _find_state(self, request: Request, proc: Process) -> _RequestState | None:
        with self._state_lock:
            for state in self.watch_con_list:
                if state.request is request and state.proc is proc:
                    return state
        return None

    def _replace_worker(self) -> None:
        with self._state_lock:
            if self._shutting_down:
                return
        self._put_available_worker(self.start_new_proc())

    def _put_available_worker(self, worker: tuple[Process, Connection]) -> bool:
        proc, connection = worker
        with self._state_lock:
            shutting_down = self._shutting_down
        if shutting_down or not _is_process_alive(proc):
            _stop_process(proc, connection, _PROCESS_JOIN_TIMEOUT)
            return False
        self.procs_and_cons.put(worker)
        return True

    def _checkout_worker(self, state: _RequestState) -> tuple[Process, Connection]:
        """切断または終了要求を監視しながら、利用可能なワーカーを一つ確保する。"""

        while True:
            if state.cancelled.is_set():
                raise CancellableRequestDisconnected("HTTP接続が切断されました")
            try:
                worker = self.procs_and_cons.get(timeout=_WORKER_WAIT_TIMEOUT)
            except queue.Empty:
                with self._state_lock:
                    if self._shutting_down:
                        raise CancellableWorkerError(
                            "キャンセル用ワーカーは終了処理中です"
                        )
                continue

            proc, connection = worker
            if not _is_process_alive(proc):
                _stop_process(proc, connection, _PROCESS_JOIN_TIMEOUT)
                self._replace_worker()
                continue

            with self._state_lock:
                cancelled = (
                    self._shutting_down or state.cancelled.is_set() or state.finalized
                )
                if not cancelled:
                    state.proc = proc
                    state.connection = connection
                    return worker
            # 取得直後にキャンセルされた場合、未使用の生存ワーカーだけを戻す。
            self._put_available_worker(worker)
            if self._shutting_down:
                raise CancellableWorkerError("キャンセル用ワーカーは終了処理中です")
            raise CancellableRequestDisconnected("HTTP接続が切断されました")

    def _finish_claimed_state(
        self,
        state: _RequestState,
        connection: Connection | None,
        *,
        reusable: bool,
    ) -> None:
        """後処理の担当が確定したワーカーを、状態に応じて再利用または交換する。"""

        proc = state.proc
        if proc is None:
            return
        connection = connection or state.connection
        with self._state_lock:
            can_reuse = (
                reusable
                and not state.cancelled.is_set()
                and not self._shutting_down
                and connection is not None
                and _is_process_alive(proc)
            )
            if can_reuse:
                self.procs_and_cons.put((proc, connection))
                return

        _stop_process(proc, connection, _PROCESS_JOIN_TIMEOUT)
        self._replace_worker()

    def finalize_con(
        self,
        req: Request,
        proc: Process,
        sub_proc_con: Connection | None,
    ) -> None:
        """リクエスト状態を一度だけ確定し、正常なワーカーは再利用、通信不能なワーカーは交換する。"""

        state = self._find_state(req, proc)
        if state is None or not self._claim_state(state):
            return
        self._finish_claimed_state(
            state,
            sub_proc_con,
            reusable=sub_proc_con is not None,
        )

    def _synthesis_impl(
        self,
        query: AudioQuery,
        speaker_id: int,
        request: Request,
        enable_interrogative_upspeak: bool,
        core_version: str | None,
    ) -> str:
        """ワーカーへ合成を委譲し、生成された一時WAVのパスを返す。

        クライアント切断を監視するため通常の合成器と異なり`Request`を受け取る。返却後の一時ファイルは呼び出し元が削除する。

        Parameters
        ----------
        query: AudioQuery
            音声合成用のクエリ
        speaker_id: int
            話者スタイルID
        request: Request
            切断を監視するHTTPリクエスト
        enable_interrogative_upspeak: bool
            疑問文の語尾を自動調整するか
        core_version: str | None
            使用するCoreのバージョン。Noneの場合は最新バージョンを使用する

        Returns
        -------
        str
            生成された一時WAVのパス
        """
        state = self._register_request(request)
        proc: Process | None = None
        sub_proc_con1: Connection | None = None
        try:
            proc, sub_proc_con1 = self._checkout_worker(state)
            sub_proc_con1.send(
                (
                    query,
                    speaker_id,
                    enable_interrogative_upspeak,
                    core_version,
                )
            )
            f_name = _receive_worker_response(sub_proc_con1)
        except CancellableRequestDisconnected:
            self._claim_state(state)
            raise
        except (EOFError, OSError) as error:
            if proc is not None:
                self.finalize_con(request, proc, None)
            else:
                self._claim_state(state)
            if state.cancelled.is_set():
                raise CancellableRequestDisconnected(
                    "HTTP接続が切断されました"
                ) from error
            raise CancellableWorkerError(
                "キャンセル用ワーカーとの通信に失敗しました"
            ) from error
        except CancellableWorkerError as error:
            if proc is not None:
                # 合成失敗後も健全なワーカーだけをプールへ戻す。
                self.finalize_con(
                    request,
                    proc,
                    sub_proc_con1 if error.reusable else None,
                )
            else:
                self._claim_state(state)
            raise
        except Exception:
            if proc is not None:
                self.finalize_con(request, proc, sub_proc_con1)
            else:
                self._claim_state(state)
            raise

        self.finalize_con(request, proc, sub_proc_con1)
        return f_name

    def _cancel_state(self, state: _RequestState) -> bool:
        with self._state_lock:
            if state.finalized:
                return False
            state.cancelled.set()
        return self._claim_state(state)

    async def _finalize_disconnected(self, state: _RequestState) -> None:
        if not self._cancel_state(state):
            return
        # terminate/joinはイベントループを止めないよう、後処理全体をスレッドへ移す。
        await asyncio.to_thread(
            self._finish_claimed_state,
            state,
            None,
            reusable=False,
        )

    async def catch_disconnection(self):
        """待機中を含む接続を監視し、切断されたリクエストをキャンセルする。"""

        while not self._shutting_down:
            await asyncio.sleep(self._disconnection_poll_interval)
            with self._state_lock:
                states = list(self.watch_con_list)
            for state in states:
                if state.finalized or not await state.request.is_disconnected():
                    continue
                await self._finalize_disconnected(state)

    def shutdown(self, timeout: float = 5.0) -> None:
        """全ワーカーを期限内に停止する。二度目以降の呼び出しは何もしない。"""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
            workers = []
            for state in self.watch_con_list:
                state.cancelled.set()
                state.finalized = True
                if state.proc is not None:
                    workers.append((state.proc, state.connection))
            self.watch_con_list.clear()
            while True:
                try:
                    workers.append(self.procs_and_cons.get_nowait())
                except queue.Empty:
                    break

        for proc, connection in workers:
            _stop_process(
                proc,
                connection,
                max(0.0, deadline - time.monotonic()),
                graceful=True,
            )

    async def shutdown_async(self, timeout: float = 5.0) -> None:
        """イベントループを止めずに、明示的なワーカー終了処理を行う。"""

        await asyncio.to_thread(self.shutdown, timeout)


def start_synthesis_subprocess(
    args: argparse.Namespace,
    sub_proc_con: Connection,
):
    """Pipeから要求を受け取り続ける音声合成ワーカーのエントリーポイント。

    spawn起動時にpickle可能なターゲットとするため、モジュール直下の関数として定義する。

    Parameters
    ----------
    args : argparse.Namespace
        デバイスやモデル探索先を含むEngine起動引数
    sub_proc_con : Connection
        親プロセスと通信する双方向Pipe
    """

    device = resolve_device(
        device=getattr(args, "device", None),
        use_gpu=getattr(args, "use_gpu", None),
    )
    # キャンセル用ワーカーも親プロセスと同じ物理デバイスを選ぶため、両方の番号を引き継ぐ。
    # キャンセル用プロセスでも通常HTTP経路と同じCOEIROINKアダプターを生成する。
    synthesis_engines = make_synthesis_engines(
        device=device,
        device_index=getattr(args, "device_index", 0),
        opencl_platform_index=getattr(args, "opencl_platform_index", 0),
        resampler=getattr(args, "resampler", "resampy"),
        voicelib_dirs=args.voicelib_dir,
        voicevox_dir=args.voicevox_dir,
        runtime_dirs=args.runtime_dir,
        cpu_num_threads=args.cpu_num_threads,
        speaker_info_dir=getattr(args, "speaker_info_dir", None),
        enable_mock=args.enable_mock,
        # 各ワーカーで全モデルを複製するとプロセス数に比例してメモリを消費するため、キャンセル経路は要求されたモデルだけを保持する。
        max_loaded_models=1,
        generator_only=getattr(args, "generator_only", False),
    )
    if not synthesis_engines:
        raise RuntimeError("音声合成エンジンがありません。")
    latest_core_version = max(synthesis_engines, key=_version_key)
    try:
        while True:
            try:
                request = sub_proc_con.recv()
            except (EOFError, OSError):
                break
            if isinstance(request, dict) and request.get("kind") == "shutdown":
                break

            temporary_wave_path: str | None = None
            try:
                query, speaker_id, enable_interrogative_upspeak, core_version = request
                if core_version is None:
                    _engine = synthesis_engines[latest_core_version]
                elif core_version in synthesis_engines:
                    _engine = synthesis_engines[core_version]
                else:
                    sub_proc_con.send({"kind": "result", "path": ""})
                    continue
                wave = _engine.synthesis(
                    query,
                    speaker_id,
                    enable_interrogative_upspeak=enable_interrogative_upspeak,
                )
                # Windowsで一時ファイルを別ハンドルから再オープンせず、作成済みハンドルへ直接書き込む。
                with NamedTemporaryFile(delete=False) as f:
                    temporary_wave_path = f.name
                    soundfile.write(
                        file=f,
                        data=wave,
                        samplerate=query.outputSamplingRate,
                        format="WAV",
                    )
                sub_proc_con.send({"kind": "result", "path": temporary_wave_path})
                # 送信成功後はHTTP応答側へ一時ファイルの所有権を移す。
                temporary_wave_path = None
            # 合成器は任意の例外を送信可能な診断情報へ変換する境界なので、ここでは広く捕捉する。
            except Exception as error:  # noqa: BLE001
                if temporary_wave_path is not None:
                    delete_file(temporary_wave_path)
                try:
                    sub_proc_con.send(_child_error_payload(error))
                except (EOFError, OSError, ValueError):
                    break
    finally:
        sub_proc_con.close()
