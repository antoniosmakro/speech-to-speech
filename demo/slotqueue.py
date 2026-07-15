"""
In-server conversation slot + waiting queue (single-container deploys).

Implements the same API contract as the Hugging Face load balancer that the
frontend already speaks, against this container's own pipeline:

  POST   /api/session      -> grant | {state:"queued", queue_id, position,
                              poll_interval_s} | 503 {state:"at_capacity"}
  GET    /api/queue/{id}   -> {state:"queued", position} | grant | 404 expired
  DELETE /api/queue/{id}   -> leave (explicit button)
  POST   /api/queue/end    -> leave (sendBeacon on teardown)

A grant's connect_url points at this container's public /v1/realtime with a
session_token; s2sproxy validates the token, binds the slot to the socket,
and releases it on disconnect — so slot lifetime is exact. Grants carry
limited:false, so the client never sends /api/session/heartbeat or /end;
the socket close IS the release signal.
State is in-memory by design: one replica owns one pipeline; a restart
expires tickets and the frontend recovers on its own. Every caller is an
async handler on the single event loop (no awaits around mutations), so the
plain dicts need no locking.

Enabled by QUEUE_ENABLED=1 (server.py then reports lb:true and delegates the
session/queue routes here). QUEUE_MAX callers may wait; beyond that,
at_capacity tells them to come back later.
"""

import logging
import os
import time
import uuid
from collections import OrderedDict

logger = logging.getLogger("s2s.slotqueue")

ENABLED = os.environ.get("QUEUE_ENABLED", "0") == "1"
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "10"))
SLOTS = int(os.environ.get("NUM_PIPELINES", "1"))
POLL_INTERVAL_S = 2
# Grant must be claimed (socket opened) within this window.
PENDING_TIMEOUT_S = int(os.environ.get("PENDING_TIMEOUT_S", "25"))
# A ticket whose owner stopped polling is presumed abandoned.
TICKET_TTL_S = 15

# session_token -> {"state": "pending"|"active", "since": ts}
_sessions: dict[str, dict] = {}
# queue_id -> {"created": ts, "last_poll": ts}
_tickets: "OrderedDict[str, dict]" = OrderedDict()


def _now() -> float:
    return time.monotonic()


def _sweep() -> None:
    """Lazily drop unclaimed grants and abandoned tickets."""
    now = _now()
    for token, s in list(_sessions.items()):
        if s["state"] == "pending" and now - s["since"] > PENDING_TIMEOUT_S:
            logger.info("grant %s unclaimed — released", token[:8])
            del _sessions[token]
    for qid, t in list(_tickets.items()):
        if now - t["last_poll"] > TICKET_TTL_S:
            logger.info("ticket %s abandoned — dropped", qid[:8])
            del _tickets[qid]


def _slot_free() -> bool:
    return len(_sessions) < SLOTS


def _grant() -> dict:
    token = uuid.uuid4().hex
    _sessions[token] = {"state": "pending", "since": _now()}
    # start.sh pins the deploy's public realtime URL (wss://<SPACE_HOST>/... on
    # a Space, ws://localhost:7860/... locally) — reuse it as the connect target.
    base = os.environ.get("SPEECH_TO_SPEECH_URL", "").strip() or "ws://localhost:7860/v1/realtime"
    logger.info("slot granted (%s pending)", token[:8])
    return {
        "session_id": token,
        "connect_url": f"{base}?session_token={token}",
        "pending_timeout_s": PENDING_TIMEOUT_S,
        "tier": "free",
        "limited": False,
    }


def _position(queue_id: str) -> int:
    for i, qid in enumerate(_tickets, start=1):
        if qid == queue_id:
            return i
    return -1


# ── Route handlers (server.py delegates here when ENABLED) ──────────────────

def create_session() -> tuple[dict, int]:
    """POST /api/session — a grant, a ticket, or at_capacity."""
    _sweep()
    if _slot_free() and not _tickets:
        return _grant(), 200
    if len(_tickets) >= QUEUE_MAX:
        return {"state": "at_capacity"}, 503
    qid = uuid.uuid4().hex
    _tickets[qid] = {"created": _now(), "last_poll": _now()}
    logger.info("queued ticket %s at position %d", qid[:8], len(_tickets))
    return {
        "state": "queued",
        "queue_id": qid,
        "position": len(_tickets),
        "poll_interval_s": POLL_INTERVAL_S,
    }, 200


def poll_queue(queue_id: str) -> tuple[dict, int]:
    """GET /api/queue/{id} — position, a grant at the front, or expired."""
    _sweep()
    ticket = _tickets.get(queue_id)
    if ticket is None:
        return {"state": "expired"}, 404
    ticket["last_poll"] = _now()
    if _slot_free() and next(iter(_tickets)) == queue_id:
        del _tickets[queue_id]
        return _grant(), 200
    return {"state": "queued", "position": _position(queue_id)}, 200


def leave_queue(queue_id: str) -> None:
    _tickets.pop(queue_id, None)


# ── Socket binding (called by s2sproxy) ─────────────────────────────────────

def claim(token: str) -> bool:
    """A realtime socket presented this token: pending -> active."""
    _sweep()
    s = _sessions.get(token or "")
    if s is None or s["state"] != "pending":
        return False
    s["state"] = "active"
    s["since"] = _now()
    logger.info("session %s active", token[:8])
    return True


def release(token: str) -> None:
    """The bound socket closed: free the slot."""
    if _sessions.pop(token or "", None):
        logger.info("session %s released", token[:8])
