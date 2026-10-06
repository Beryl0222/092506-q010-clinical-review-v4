"""SQLite 持久化边界。

- 单连接 + 进程内可重入锁，所有写操作在事务中完成，保证并发审核时
  状态机的条件更新（CAS）不会互相覆盖。
- 状态推进统一使用 ``UPDATE ... WHERE status = ?`` 形式的 CAS，
  影响行数为 0 即表示已被其他审核者抢先处理。
"""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from .domain import (
    AnswerVersion,
    CallbackRecord,
    Case,
    Event,
    Evidence,
    Notification,
    Record,
    SessionRecord,
)


def _locked(method):
    """只读方法同样需要经过连接锁：多线程共用一个 sqlite 连接。"""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        # check_same_thread=False：由 self._lock 保证串行访问
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        self._create_schema()

    # -- 事务 ------------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    @contextmanager
    def transaction(self):
        """可重入事务：最外层提交，异常时整体回滚。"""
        with self._lock:
            outer = self._depth == 0
            if outer:
                self.connection.execute("BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
                if outer:
                    self.connection.commit()
            except Exception:
                if outer:
                    self.connection.rollback()
                raise
            finally:
                self._depth -= 1

    def _create_schema(self) -> None:
        with self.transaction():
            conn = self.connection
            # 基线表，保持原有登记能力
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_case (
                    record_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_cases (
                    case_id TEXT PRIMARY KEY,
                    dedup_key TEXT NOT NULL UNIQUE,
                    question_safe TEXT NOT NULL,
                    status TEXT NOT NULL,
                    session_id TEXT,
                    current_version_id TEXT,
                    require_doctor INTEGER NOT NULL DEFAULT 0,
                    claimed_by TEXT,
                    stale INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS answer_versions (
                    version_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    answer_text TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    risk_tags TEXT NOT NULL,
                    route TEXT NOT NULL,
                    require_doctor INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    reasons TEXT NOT NULL DEFAULT '[]',
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT NOT NULL DEFAULT '',
                    UNIQUE(case_id, seq)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confirmations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    doctor_id TEXT,
                    reviewer_id TEXT,
                    kind TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    comment TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    version_id TEXT,
                    kind TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_callbacks (
                    callback_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    version_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    closed_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notifications (
                    notification_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    version_id TEXT,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    delivered_at TEXT NOT NULL DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidences (
                    evidence_id TEXT PRIMARY KEY,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    superseded_by TEXT,
                    title TEXT NOT NULL DEFAULT ''
                )
            """)
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_versions_case ON answer_versions(case_id)",
                "CREATE INDEX IF NOT EXISTS idx_versions_status ON answer_versions(status)",
                "CREATE INDEX IF NOT EXISTS idx_cases_status ON review_cases(status)",
                "CREATE INDEX IF NOT EXISTS idx_events_case ON review_events(case_id)",
                "CREATE INDEX IF NOT EXISTS idx_callbacks_status ON model_callbacks(status)",
                "CREATE INDEX IF NOT EXISTS idx_notif_session ON notifications(session_id, delivered_at)",
            ):
                conn.execute(stmt)

    # -- 基线 ------------------------------------------------------------

    def save(self, record: Record) -> Record:
        value = record.with_timestamp()
        with self.transaction():
            self.connection.execute(
                "INSERT INTO review_case(record_id, owner_id, state, created_at)"
                " VALUES(?,?,?,?)",
                (value.record_id, value.owner_id, value.state, value.created_at),
            )
        return value

    @_locked
    def get(self, record_id: str) -> Record | None:
        row = self.connection.execute(
            "SELECT record_id, owner_id, state, created_at FROM review_case"
            " WHERE record_id=?",
            (record_id,),
        ).fetchone()
        return Record(**dict(row)) if row else None

    # -- cases -----------------------------------------------------------

    def insert_case(self, case: Case) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO review_cases(case_id, dedup_key, question_safe,"
                " status, session_id, current_version_id, require_doctor,"
                " claimed_by, stale, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (case.case_id, case.dedup_key, case.question_safe,
                 case.status, case.session_id, case.current_version_id,
                 int(case.require_doctor), case.claimed_by, int(case.stale),
                 case.created_at, case.updated_at),
            )

    @_locked
    def get_case(self, case_id: str) -> Case | None:
        row = self.connection.execute(
            "SELECT * FROM review_cases WHERE case_id=?", (case_id,)
        ).fetchone()
        return self._case_row(row) if row else None

    @_locked
    def get_case_by_dedup(self, dedup_key: str) -> Case | None:
        row = self.connection.execute(
            "SELECT * FROM review_cases WHERE dedup_key=?", (dedup_key,)
        ).fetchone()
        return self._case_row(row) if row else None

    @staticmethod
    def _case_row(row: sqlite3.Row) -> Case:
        return Case(
            case_id=row["case_id"], dedup_key=row["dedup_key"],
            question_safe=row["question_safe"], status=row["status"],
            session_id=row["session_id"],
            current_version_id=row["current_version_id"],
            require_doctor=bool(row["require_doctor"]),
            claimed_by=row["claimed_by"], stale=bool(row["stale"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    @_locked
    def list_cases(self, statuses: tuple[str, ...] | None = None) -> list[Case]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.connection.execute(
                f"SELECT * FROM review_cases WHERE status IN ({placeholders})"
                " ORDER BY created_at", statuses,
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM review_cases ORDER BY created_at"
            ).fetchall()
        return [self._case_row(row) for row in rows]

    def cas_case(self, case_id: str, expected_statuses: tuple[str, ...],
                 changes: dict, expected_unclaimed: bool = False) -> bool:
        """条件更新案件；状态不匹配或已被认领则失败。"""
        sets = ", ".join(f"{key}=?" for key in changes)
        where = "case_id=? AND status IN (" \
            + ",".join("?" for _ in expected_statuses) + ")"
        params = list(changes.values()) + [case_id] + list(expected_statuses)
        if expected_unclaimed:
            where += " AND claimed_by IS NULL"
        with self.transaction():
            cur = self.connection.execute(
                f"UPDATE review_cases SET {sets} WHERE {where}",
                params,
            )
            return cur.rowcount == 1

    # -- versions --------------------------------------------------------

    def insert_version(self, version: AnswerVersion) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO answer_versions(version_id, case_id, seq,"
                " answer_text, citations, risk_tags, route, require_doctor,"
                " status, reasons, created_by, created_at, decided_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (version.version_id, version.case_id, version.seq,
                 version.answer_text, json.dumps(list(version.citations)),
                 json.dumps(list(version.risk_tags)), version.route,
                 int(version.require_doctor), version.status,
                 json.dumps(list(version.reasons)), version.created_by,
                 version.created_at, version.decided_at),
            )

    @_locked
    def get_version(self, version_id: str) -> AnswerVersion | None:
        row = self.connection.execute(
            "SELECT * FROM answer_versions WHERE version_id=?", (version_id,)
        ).fetchone()
        return self._version_row(row) if row else None

    @_locked
    def list_versions(self, case_id: str) -> list[AnswerVersion]:
        rows = self.connection.execute(
            "SELECT * FROM answer_versions WHERE case_id=? ORDER BY seq",
            (case_id,),
        ).fetchall()
        return [self._version_row(row) for row in rows]

    @_locked
    def list_versions_by_status(self, statuses: tuple[str, ...]) -> list[AnswerVersion]:
        placeholders = ",".join("?" for _ in statuses)
        rows = self.connection.execute(
            f"SELECT * FROM answer_versions WHERE status IN ({placeholders})",
            statuses,
        ).fetchall()
        return [self._version_row(row) for row in rows]

    @staticmethod
    def _version_row(row: sqlite3.Row) -> AnswerVersion:
        return AnswerVersion(
            version_id=row["version_id"], case_id=row["case_id"],
            seq=row["seq"], answer_text=row["answer_text"],
            citations=tuple(json.loads(row["citations"])),
            risk_tags=tuple(json.loads(row["risk_tags"])),
            route=row["route"], require_doctor=bool(row["require_doctor"]),
            status=row["status"],
            reasons=tuple(json.loads(row["reasons"] or "[]")),
            created_by=row["created_by"], created_at=row["created_at"],
            decided_at=row["decided_at"],
        )

    def cas_version(self, version_id: str, expected_statuses: tuple[str, ...],
                    changes: dict) -> bool:
        sets = ", ".join(f"{key}=?" for key in changes)
        where = "version_id=? AND status IN (" \
            + ",".join("?" for _ in expected_statuses) + ")"
        params = list(changes.values()) + [version_id] \
            + list(expected_statuses)
        with self.transaction():
            cur = self.connection.execute(
                f"UPDATE answer_versions SET {sets} WHERE {where}", params,
            )
            return cur.rowcount == 1

    def next_seq(self, case_id: str) -> int:
        with self.transaction():
            row = self.connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq"
                " FROM answer_versions WHERE case_id=?",
                (case_id,),
            ).fetchone()
            return int(row["next_seq"])

    # -- confirmations ---------------------------------------------------

    def add_confirmation(self, version_id: str, case_id: str, actor: str,
                         kind: str, created_at: str,
                         doctor_id: str = "", reviewer_id: str = "",
                         comment: str = "") -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO confirmations(version_id, case_id, doctor_id,"
                " reviewer_id, kind, actor, comment, created_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (version_id, case_id, doctor_id, reviewer_id, kind, actor,
                 comment, created_at),
            )

    @_locked
    def has_confirmation(self, version_id: str, kind: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM confirmations WHERE version_id=? AND kind=? LIMIT 1",
            (version_id, kind),
        ).fetchone()
        return row is not None

    # -- events ----------------------------------------------------------

    def insert_event(self, event: Event) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO review_events(event_id, case_id, version_id,"
                " kind, actor, payload, created_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (event.event_id, event.case_id, event.version_id, event.kind,
                 event.actor, event.payload, event.created_at),
            )

    @_locked
    def list_events(self, case_id: str) -> list[Event]:
        rows = self.connection.execute(
            "SELECT * FROM review_events WHERE case_id=? ORDER BY created_at, rowid",
            (case_id,),
        ).fetchall()
        return [Event(**dict(row)) for row in rows]

    # -- callbacks -------------------------------------------------------

    def insert_callback_if_absent(
            self, callback: CallbackRecord) -> tuple[CallbackRecord, bool]:
        """幂等登记延迟回调：重复 callback_id 返回 ``(既有记录, False)``。"""
        with self.transaction():
            existing = self.connection.execute(
                "SELECT * FROM model_callbacks WHERE callback_id=?",
                (callback.callback_id,),
            ).fetchone()
            if existing:
                return CallbackRecord(
                    callback_id=existing["callback_id"],
                    case_id=existing["case_id"],
                    version_id=existing["version_id"],
                    status=existing["status"],
                    created_at=existing["created_at"],
                    completed_at=existing["completed_at"],
                ), False
            self.connection.execute(
                "INSERT INTO model_callbacks(callback_id, case_id, version_id,"
                " status, created_at, completed_at) VALUES(?,?,?,?,?,?)",
                (callback.callback_id, callback.case_id, callback.version_id,
                 callback.status, callback.created_at, callback.completed_at),
            )
            return callback, True

    @_locked
    def get_callback(self, callback_id: str) -> CallbackRecord | None:
        row = self.connection.execute(
            "SELECT * FROM model_callbacks WHERE callback_id=?", (callback_id,)
        ).fetchone()
        if not row:
            return None
        return CallbackRecord(
            callback_id=row["callback_id"], case_id=row["case_id"],
            version_id=row["version_id"], status=row["status"],
            created_at=row["created_at"], completed_at=row["completed_at"],
        )

    def cas_callback(self, callback_id: str, completed_at: str) -> bool:
        """pending -> done 的 CAS，重复/延迟回调第二次返回 False。"""
        with self.transaction():
            cur = self.connection.execute(
                "UPDATE model_callbacks SET status='done', completed_at=?"
                " WHERE callback_id=? AND status='pending'",
                (completed_at, callback_id),
            )
            return cur.rowcount == 1

    @_locked
    def list_pending_callbacks(self) -> list[CallbackRecord]:
        rows = self.connection.execute(
            "SELECT * FROM model_callbacks WHERE status='pending'"
        ).fetchall()
        return [CallbackRecord(
            callback_id=row["callback_id"], case_id=row["case_id"],
            version_id=row["version_id"], status=row["status"],
            created_at=row["created_at"], completed_at=row["completed_at"],
        ) for row in rows]

    # -- evidences -------------------------------------------------------

    def upsert_evidence(self, evidence: Evidence) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO evidences(evidence_id, valid_from, valid_until,"
                " superseded_by, title) VALUES(?,?,?,?,?)"
                " ON CONFLICT(evidence_id) DO UPDATE SET"
                " valid_until=excluded.valid_until,"
                " superseded_by=excluded.superseded_by, title=excluded.title",
                (evidence.evidence_id, evidence.valid_from,
                 evidence.valid_until, evidence.superseded_by,
                 evidence.title),
            )

    @_locked
    def list_evidences(self) -> list[Evidence]:
        rows = self.connection.execute(
            "SELECT * FROM evidences ORDER BY evidence_id"
        ).fetchall()
        return [Evidence(evidence_id=row["evidence_id"],
                         valid_from=row["valid_from"],
                         valid_until=row["valid_until"],
                         superseded_by=row["superseded_by"],
                         title=row["title"]) for row in rows]

    # -- sessions --------------------------------------------------------

    def upsert_session(self, session: SessionRecord) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO sessions(session_id, status, created_at, closed_at)"
                " VALUES(?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET"
                " status=excluded.status, closed_at=excluded.closed_at",
                (session.session_id, session.status, session.created_at,
                 session.closed_at),
            )

    @_locked
    def get_session(self, session_id: str) -> SessionRecord | None:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        return SessionRecord(session_id=row["session_id"], status=row["status"],
                             created_at=row["created_at"],
                             closed_at=row["closed_at"])

    @_locked
    def list_active_sessions(self) -> list[SessionRecord]:
        rows = self.connection.execute(
            "SELECT * FROM sessions WHERE status='active'"
        ).fetchall()
        return [SessionRecord(session_id=row["session_id"],
                              status=row["status"],
                              created_at=row["created_at"],
                              closed_at=row["closed_at"]) for row in rows]

    # -- notifications ---------------------------------------------------

    def insert_notification(self, notification: Notification) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO notifications(notification_id, session_id,"
                " case_id, version_id, kind, payload, created_at, delivered_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (notification.notification_id, notification.session_id,
                 notification.case_id, notification.version_id,
                 notification.kind, notification.payload,
                 notification.created_at, notification.delivered_at),
            )
    @_locked
    def list_undelivered(self, session_id: str | None = None) -> list[Notification]:
        if session_id is None:
            rows = self.connection.execute(
                "SELECT * FROM notifications WHERE delivered_at=''"
                " ORDER BY created_at, rowid"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT * FROM notifications WHERE session_id=? AND delivered_at=''"
                " ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [Notification(**dict(row)) for row in rows]

    def mark_delivered(self, notification_id: str, delivered_at: str) -> bool:
        with self.transaction():
            cur = self.connection.execute(
                "UPDATE notifications SET delivered_at=?"
                " WHERE notification_id=? AND delivered_at=''",
                (delivered_at, notification_id),
            )
            return cur.rowcount == 1

    def close(self) -> None:
        with self._lock:
            self.connection.close()
