"""Session lifecycle management.

Sessions are in-memory, bounded, isolated per client, and expired on a TTL.
Frame buffers are released on every terminal path (completion, error,
cancellation, expiry) so memory stays bounded.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SessionState(str, Enum):
    CREATED = "created"
    STREAMING = "streaming"
    INFERRING = "inferring"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class SessionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Session:
    session_id: str
    ws_token: str
    num_frames: int
    max_frame_bytes: int
    max_total_frames: int
    ttl_seconds: int
    created_at: float = field(default_factory=time.time)
    state: SessionState = SessionState.CREATED
    frames: Dict[int, bytes] = field(default_factory=dict)
    next_expected_index: int = 0
    # True once client declares it has sent everything.
    client_complete: bool = False
    error: Optional[SessionError] = None
    # Guard against a second concurrent inference for the same session.
    inference_started: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def received_count(self) -> int:
        return len(self.frames)

    @property
    def is_complete(self) -> bool:
        # Auto-finalize once every required frame has arrived. The explicit
        # "complete" control message is an early-validation shortcut that lets a
        # client detect an incomplete sequence without streaming all frames.
        return len(self.frames) >= self.num_frames

    @property
    def remaining_ttl(self) -> float:
        return max(0.0, self.ttl_seconds - (time.time() - self.created_at))

    def ordered_frames(self) -> List[bytes]:
        """Return frames ordered by sequence index."""
        return [self.frames[i] for i in sorted(self.frames)]


class SessionManager:
    def __init__(self, max_sessions: int, ttl_seconds: int):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds

    def create(self, num_frames: int, max_frame_bytes: int, max_total_frames: int) -> Session:
        with self._lock:
            self._evict_expired_locked()
            if len(self._sessions) >= self._max_sessions:
                raise SessionError("server_busy", "Too many active sessions; try again shortly")
            session = Session(
                session_id=secrets.token_urlsafe(32),
                ws_token=secrets.token_urlsafe(32),
                num_frames=num_frames,
                max_frame_bytes=max_frame_bytes,
                max_total_frames=max_total_frames,
                ttl_seconds=self._ttl_seconds,
            )
            self._sessions[session.session_id] = session
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            self._evict_expired_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise SessionError("not_found", "Session not found or expired")
            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_count(self) -> int:
        with self._lock:
            self._evict_expired_locked()
            return len(self._sessions)

    def evict_expired(self) -> int:
        with self._lock:
            return self._evict_expired_locked()

    def _evict_expired_locked(self) -> int:
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.created_at > s.ttl_seconds and s.state not in (
                SessionState.COMPLETED, SessionState.FAILED, SessionState.CANCELLED
            )
        ]
        for sid in expired:
            s = self._sessions[sid]
            s.state = SessionState.EXPIRED
            s.frames.clear()
        # Also drop long-finished sessions to keep the table small.
        finished = [
            sid for sid, s in self._sessions.items()
            if s.state in (SessionState.COMPLETED, SessionState.FAILED,
                           SessionState.CANCELLED, SessionState.EXPIRED)
        ]
        for sid in finished:
            self._sessions[sid].frames.clear()
            del self._sessions[sid]
        return len(expired)
