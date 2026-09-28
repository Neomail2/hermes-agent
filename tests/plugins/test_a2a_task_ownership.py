"""A2A: a final belongs to the exact task that owns it (reply anchor == task id), never the oldest
pending task of the context; delegation ACKs stay interim across waves; admission is bounded.

Loopback-only: the real adapter HTTP server, BasePlatformAdapter.handle_message, the real
tools.async_delegation dispatch/finalize and the real GatewayRunner async-delegation watcher are
exercised. Only the model turns and the child bodies are simulated.
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.request
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.adapter import A2AAdapter

WORKING = ("TASK_STATE_WORKING", "TASK_STATE_SUBMITTED")


def _bare():
    return A2AAdapter(PlatformConfig(enabled=True))


# --- exact ownership (unit level) -------------------------------------------------------------------

def test_reverse_order_anchored_finals_settle_their_own_task():
    a = _bare()
    f1, f2 = a._add_pending("task-1", "ctx"), a._add_pending("task-2", "ctx")

    async def run():
        await a.send("ctx", "reply two", reply_to="task-2", metadata={"notify": True})
        assert f2.result(timeout=0) == (protocol.STATE_COMPLETED, "reply two")
        assert not f1.done()
        await a.send("ctx", "late duplicate two", reply_to="task-2", metadata={"notify": True})
        assert not f1.done()
        await a.send("ctx", "reply one", metadata={"notify": True, "reply_to_message_id": "task-1"})
        assert f1.result(timeout=0) == (protocol.STATE_COMPLETED, "reply one")
    try:
        asyncio.run(run())
    finally:
        a._pop_pending("task-1")
        a._pop_pending("task-2")


@pytest.mark.parametrize("first_end", ["completed", "failed", "cancelled"])
def test_unanchored_late_final_never_settles_the_sole_pending_sibling(first_end):
    a = _bare()
    fa, fb = a._add_pending("task-A", "ctx"), a._add_pending("task-B", "ctx")

    async def run():
        if first_end == "completed":
            await a.send("ctx", "A final", reply_to="task-A", metadata={"notify": True})
        else:
            a._resolve_task("task-A", protocol.STATE_FAILED if first_end == "failed" else protocol.STATE_CANCELED, "")
        a._pop_pending("task-A")
        await a.send("ctx", "late A text without anchor", metadata={"notify": True})
        assert not fb.done(), "an unanchored final was assigned to the only pending task"
        await a.send("ctx", "B final", reply_to="task-B", metadata={"notify": True})
        assert fb.result(timeout=0) == (protocol.STATE_COMPLETED, "B final")
    try:
        asyncio.run(run())
    finally:
        a._pop_pending("task-A")
        a._pop_pending("task-B")


def test_admission_cap_rejects_before_dispatch(monkeypatch):
    monkeypatch.setenv("A2A_MAX_INFLIGHT", "1")
    a = _bare()
    a._loop = object()
    dispatched = []
    a._message_handler = lambda event: dispatched.append(event)
    a._add_pending("already-running", "other-ctx")
    task, pending = a._prepare_task({"message": {"parts": [{"text": "hello"}], "contextId": "ctx"}}, "peer")
    assert pending is None and task["status"]["state"] == protocol.STATE_REJECTED
    assert dispatched == []
    a._pop_pending("already-running")


def test_waiter_start_failure_fails_task_and_frees_admission_slot(monkeypatch):
    monkeypatch.setenv("A2A_MAX_INFLIGHT", "1")
    a = _bare()
    a._loop = object()
    a._message_handler = lambda event: None
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coro, loop: coro.close())
    real_start = threading.Thread.start

    def start(self):
        if self.name.startswith("a2a-wait-"):
            raise RuntimeError("can't start new thread")
        return real_start(self)
    monkeypatch.setattr(threading.Thread, "start", start)
    params = {"message": {"parts": [{"text": "hi"}], "contextId": "ctx"}, "configuration": {"returnImmediately": True}}
    task = a._rpc_message_send(1, params, "peer")["result"]
    assert task["status"]["state"] == protocol.STATE_FAILED
    assert a.tasks.get(task["id"])["state"] == protocol.STATE_FAILED
    assert not a._active_tasks and not a._pending
    monkeypatch.setattr(threading.Thread, "start", real_start)
    nxt, pending = a._prepare_task({"message": {"parts": [{"text": "again"}], "contextId": "ctx2"}}, "peer")
    assert nxt is None and pending is not None
    a._pop_pending(pending["task_id"])


@pytest.mark.parametrize("streamed", [True, False])
def test_success_settles_on_returned_or_streamed_final_but_not_on_yield(streamed):
    from gateway.platforms.event import ProcessingOutcome
    a = _bare()
    fut = a._add_pending("task-s", "ctx-s")
    event = type("E", (), {})()
    event.message_id = "task-s"
    try:
        event._a2a_handler_response = None  # yielded: no final yet
        asyncio.run(a.on_processing_complete(event, ProcessingOutcome.SUCCESS))
        assert not fut.done()
        if streamed:
            event._streamed_final_response = "streamed body"
        else:
            event._a2a_handler_response = "returned body"
        asyncio.run(a.on_processing_complete(event, ProcessingOutcome.SUCCESS))
        assert fut.result(timeout=0) == (protocol.STATE_COMPLETED, "streamed body" if streamed else "returned body")
    finally:
        a._pop_pending("task-s")


# --- process completions for two tasks of one context stay separate ---------------------------------

def _batch_runner(delivered):
    from gateway.run import GatewayRunner
    r = object.__new__(GatewayRunner)

    async def deliver(synth_text, evt):
        delivered.append((evt.get("message_id"), synth_text))
        return True
    r._deliver_completion_notification = deliver
    r._record_coalesced_completion_siblings = lambda events: None
    return r


def _completion(platform, mid, sid):
    return {"type": "completion", "session_id": sid, "session_key": f"agent:main:{platform}:dm:ctx-1",
            "platform": platform, "chat_type": "dm", "chat_id": "ctx-1", "thread_id": "", "user_id": "peer",
            "message_id": mid, "exit_code": 0, "output": "RESULT-" + sid, "command": "x"}


def _enqueue_two(platform):
    delivered = []

    async def main():
        r = _batch_runner(delivered)
        return await asyncio.gather(
            r._enqueue_process_completion_notification("done A", _completion(platform, "task-A", "procA")),
            r._enqueue_process_completion_notification("done B", _completion(platform, "task-B", "procB")))
    return asyncio.run(main()), delivered


def test_a2a_process_completions_of_two_tasks_keep_their_own_anchor():
    results, delivered = _enqueue_two("a2a")
    assert results == [True, True]
    assert sorted(mid for mid, _ in delivered) == ["task-A", "task-B"], delivered


def test_non_a2a_process_completions_still_coalesce():
    results, delivered = _enqueue_two("telegram")
    assert results == [True, True] and len(delivered) == 1


@pytest.mark.parametrize("platform,groups", [("a2a", 2), ("telegram", 1)])
def test_async_delegation_group_key_splits_by_anchor_for_a2a_only(platform, groups):
    from gateway.run import GatewayRunner
    evts = [dict(_completion(platform, mid, "p"), type="async_delegation", parent_session_id="p")
            for mid in ("task-A", "task-B")]
    assert len({GatewayRunner._async_group_key(e) for e in evts}) == groups


# --- loopback: returnImmediately + GetTask across delegation waves ----------------------------------

@pytest.fixture
def loop_env(monkeypatch, tmp_path):
    for k in ("A2A_BEARER_TOKEN", "A2A_PEER_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("A2A_PORT", str(port))
    import agent.oneshot_footprint as footprint
    monkeypatch.setattr(footprint, "is_single_query_session", lambda: False)
    import tools.async_delegation as ad
    ad._reset_for_tests()
    yield port
    ad._reset_for_tests()


def _rpc(port, method, params):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as r:
        return json.load(r).get("result")


def _wait(pred, timeout=8.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _gateway_runner(adapter):
    from gateway.run import GatewayRunner
    r = object.__new__(GatewayRunner)
    r._running = True
    r.adapters = {}
    r._session_source_cache = {}
    r._completion_delivery_lock = threading.Lock()
    r._completion_deliveries_inflight = set()
    r._completion_deliveries_delivered = OrderedDict()
    r._completion_delivery_retention = 2048
    r._background_tasks = set()

    async def get_session(sid):
        return {"id": sid, "ended_at": None}
    r._session_db = SimpleNamespace(get_session=get_session)
    r._build_process_event_source = lambda evt: adapter.build_source(
        chat_id=evt["chat_id"], chat_type="dm", user_id=evt.get("user_id"), user_name=evt.get("user_name"))
    r._resolve_injection_adapter = lambda platform_name, source=None: adapter if platform_name == "a2a" else None
    return r


def _dispatch(event, label, gate=None):
    """What a turn calling delegate_task(background=True) does, with a synthetic child body."""
    import tools.async_delegation as ad
    from gateway.session_context import clear_session_vars, set_session_vars
    ctx = event.source.chat_id
    tok = set_session_vars(platform="a2a", chat_id=ctx, chat_type="dm", session_key="agent:main:a2a:dm:" + ctx,
                           message_id=str(event.message_id or ""))
    try:
        res = ad.dispatch_async_delegation(
            goal=label, context=None, toolsets=None, role="leaf", model="m",
            session_key="agent:main:a2a:dm:" + ctx, parent_session_id="sess-" + ctx,
            runner=lambda: (gate.wait(10) if gate else None, {"status": "completed", "summary": "RESULT-" + label})[1])
        assert res["status"] == "dispatched", res
        if gate is None:  # fast child: finished before this turn returns its ACK
            assert _wait(lambda: (ad._records.get(res["delegation_id"]) or {}).get("status") not in ad._LIVE_STATES)
    finally:
        clear_session_vars(tok)


@pytest.mark.parametrize("plan", [(), ("fast",), ("fast", "fast"), ("slow", "fast")],
                         ids=lambda p: "-".join(p) or "single-wave")
def test_return_immediately_final_only_after_last_wave(loop_env, plan):
    port = loop_env
    a = A2AAdapter(PlatformConfig(enabled=True, extra={"port": port}))
    wakes, gates = [], []

    async def model(event):
        if not getattr(event, "internal", False):
            _dispatch(event, "W0")  # fast first child
            return "ACK 0 NOT FINAL"
        wakes.append(event.message_id)
        n = len(wakes)
        if n > len(plan):
            return "FINAL AFTER %d WAKES" % n
        gate = threading.Event() if plan[n - 1] == "slow" else None
        if gate:
            gates.append(gate)
        _dispatch(event, "W%d" % n, gate)
        return "ACK %d NOT FINAL" % n
    a._message_handler = model

    def body():
        tid = _rpc(port, "SendMessage", {"message": {"messageId": "m-1", "role": "ROLE_USER", "contextId": "ctx-w",
                                                     "parts": [{"text": "work"}]},
                                         "configuration": {"returnImmediately": True}})["task"]["id"]
        first = _rpc(port, "GetTask", {"id": tid})["status"]["state"]
        released = 0
        end = time.time() + 20
        while time.time() < end:
            while released < len(gates):
                time.sleep(0.3)
                assert _rpc(port, "GetTask", {"id": tid})["status"]["state"] in WORKING, "an ACK became final"
                gates[released].set()
                released += 1
            if _rpc(port, "GetTask", {"id": tid})["status"]["state"] not in WORKING:
                break
            time.sleep(0.05)
        return tid, first, _rpc(port, "GetTask", {"id": tid})

    async def main():
        assert await a.connect()
        runner = _gateway_runner(a)
        watcher = asyncio.ensure_future(runner._async_delegation_watcher(interval=0.05))
        try:
            return await asyncio.to_thread(body)
        finally:
            runner._running = False
            watcher.cancel()
            await a.disconnect()

    tid, first, done = asyncio.run(main())
    assert first in WORKING
    assert done["status"]["state"] == "TASK_STATE_COMPLETED", done
    assert protocol.extract_text(done["artifacts"][0]) == "FINAL AFTER %d WAKES" % (len(plan) + 1)
    assert wakes and set(wakes) == {tid}
