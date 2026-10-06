"""SQLite 持久化边界：会话、版本、证据、审核事件、通知、幂等键与回调记录。

单连接 + 可重入锁保证进程内多线程下的串行写入；跨进程/重启通过文件库恢复。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .domain import AnswerVersion, Citation, Evidence, Session, utcnow_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session (
    session_id   TEXT PRIMARY KEY,
    question     TEXT NOT NULL,
    risk_labels  TEXT NOT NULL,
    state        TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answer_version (
    version_id      TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT NOT NULL,
    state           TEXT NOT NULL,
    created_by      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    decided_by      TEXT NOT NULL DEFAULT '',
    decided_at      TEXT NOT NULL DEFAULT '',
    decision_reason TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status      TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_event (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    version_id TEXT NOT NULL DEFAULT '',
    actor      TEXT NOT NULL,
    role       TEXT NOT NULL,
    action     TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notification (
    notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    message    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency_key (
    key        TEXT PRIMARY KEY,
    scope      TEXT NOT NULL,
    result     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS callback (
    callback_id TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    applied     INTEGER NOT NULL,
    received_at TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.connection.executescript(_SCHEMA)
            self.connection.commit()

    # ---- 会话 ----
    def insert_session(self, session: Session) -> None:
        self.connection.execute(
            "INSERT INTO session(session_id, question, risk_labels, state, created_at, updated_at)"
            " VALUES(?,?,?,?,?,?)",
            (session.session_id, session.question, json.dumps(list(session.risk_labels)),
             session.state, session.created_at, session.updated_at),
        )

    def get_session(self, session_id: str) -> Session | None:
        row = self.connection.execute(
            "SELECT * FROM session WHERE session_id=?", (session_id,)).fetchone()
        return self._to_session(row) if row else None

    def update_session_state(self, session_id: str, state: str, now: str) -> None:
        self.connection.execute(
            "UPDATE session SET state=?, updated_at=? WHERE session_id=?",
            (state, now, session_id))

    def list_sessions_by_state(self, states: tuple[str, ...]) -> list[Session]:
        marks = ",".join("?" for _ in states)
        rows = self.connection.execute(
            f"SELECT * FROM session WHERE state IN ({marks}) ORDER BY created_at", states
        ).fetchall()
        return [self._to_session(r) for r in rows]

    # ---- 版本 ----
    def insert_version(self, v: AnswerVersion) -> None:
        self.connection.execute(
            "INSERT INTO answer_version(version_id, session_id, seq, content, citations, state,"
            " created_by, created_at) VALUES(?,?,?,?,?,?,?,?)",
            (v.version_id, v.session_id, v.seq, v.content,
             json.dumps([c.__dict__ for c in v.citations]), v.state, v.created_by, v.created_at))

    def get_version(self, version_id: str) -> AnswerVersion | None:
        row = self.connection.execute(
            "SELECT * FROM answer_version WHERE version_id=?", (version_id,)).fetchone()
        return self._to_version(row) if row else None

    def next_seq(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM answer_version WHERE session_id=?",
            (session_id,)).fetchone()
        return int(row["n"])

    def decide_version(self, version_id: str, state: str, actor: str, now: str, reason: str = "") -> int:
        """仅当版本仍处于待审核态时推进状态，返回受影响行数（并发守门）。"""
        cur = self.connection.execute(
            "UPDATE answer_version SET state=?, decided_by=?, decided_at=?, decision_reason=?"
            " WHERE version_id=? AND state='pending_review'",
            (state, actor, now, reason, version_id))
        return cur.rowcount

    def set_version_state(self, version_id: str, state: str, now: str, reason: str = "") -> None:
        self.connection.execute(
            "UPDATE answer_version SET state=?, decided_at=?, decision_reason=? WHERE version_id=?",
            (state, now, reason, version_id))

    def list_versions(self, session_id: str) -> list[AnswerVersion]:
        rows = self.connection.execute(
            "SELECT * FROM answer_version WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [self._to_version(r) for r in rows]

    def list_published_citing(self, evidence_id: str) -> list[AnswerVersion]:
        rows = self.connection.execute(
            "SELECT * FROM answer_version WHERE state='published'").fetchall()
        return [self._to_version(r) for r in rows
                if any(c["evidence_id"] == evidence_id for c in json.loads(r["citations"]))]

    def list_pending_citing(self, evidence_id: str) -> list[AnswerVersion]:
        rows = self.connection.execute(
            "SELECT * FROM answer_version WHERE state='pending_review'").fetchall()
        return [self._to_version(r) for r in rows
                if any(c["evidence_id"] == evidence_id for c in json.loads(r["citations"]))]

    # ---- 证据 ----
    def upsert_evidence(self, e: Evidence, now: str) -> None:
        self.connection.execute(
            "INSERT INTO evidence(evidence_id, title, valid_until, status, updated_at)"
            " VALUES(?,?,?,?,?)"
            " ON CONFLICT(evidence_id) DO UPDATE SET title=excluded.title,"
            " valid_until=excluded.valid_until, status=excluded.status, updated_at=excluded.updated_at",
            (e.evidence_id, e.title, e.valid_until, e.status, now))

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        row = self.connection.execute(
            "SELECT * FROM evidence WHERE evidence_id=?", (evidence_id,)).fetchone()
        return Evidence(row["evidence_id"], row["title"], row["valid_until"], row["status"]) if row else None

    def evidence_map(self, ids: list[str]) -> dict[str, Evidence]:
        return {i: e for i in ids if (e := self.get_evidence(i)) is not None}

    # ---- 事件 / 通知 ----
    def add_event(self, session_id: str, version_id: str, actor: str, role: str,
                  action: str, detail: str, now: str) -> None:
        self.connection.execute(
            "INSERT INTO review_event(session_id, version_id, actor, role, action, detail, created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (session_id, version_id, actor, role, action, detail, now))

    def list_events(self, session_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM review_event WHERE session_id=? ORDER BY event_id", (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def add_notification(self, session_id: str, kind: str, message: str, now: str) -> None:
        self.connection.execute(
            "INSERT INTO notification(session_id, kind, message, created_at) VALUES(?,?,?,?)",
            (session_id, kind, message, now))

    def list_notifications(self, session_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM notification WHERE session_id=? ORDER BY notification_id",
            (session_id,)).fetchall()
        return [dict(r) for r in rows]

    # ---- 幂等键与回调 ----
    def get_idempotent(self, key: str) -> dict | None:
        row = self.connection.execute(
            "SELECT result FROM idempotency_key WHERE key=?", (key,)).fetchone()
        return json.loads(row["result"]) if row else None

    def put_idempotent(self, key: str, scope: str, result: dict, now: str) -> None:
        self.connection.execute(
            "INSERT INTO idempotency_key(key, scope, result, created_at) VALUES(?,?,?,?)",
            (key, scope, json.dumps(result, ensure_ascii=False), now))

    def record_callback(self, callback_id: str, session_id: str, applied: bool, now: str) -> bool:
        """登记回调；callback_id 已存在返回 False（重复回调）。"""
        try:
            self.connection.execute(
                "INSERT INTO callback(callback_id, session_id, applied, received_at) VALUES(?,?,?,?)",
                (callback_id, session_id, 1 if applied else 0, now))
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_callback_applied(self, callback_id: str) -> None:
        self.connection.execute(
            "UPDATE callback SET applied=1 WHERE callback_id=?", (callback_id,))

    # ---- 事务 ----
    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _to_session(row: sqlite3.Row) -> Session:
        return Session(row["session_id"], row["question"], tuple(json.loads(row["risk_labels"])),
                       row["state"], row["created_at"], row["updated_at"])

    @staticmethod
    def _to_version(row: sqlite3.Row) -> AnswerVersion:
        citations = tuple(Citation(**c) for c in json.loads(row["citations"]))
        return AnswerVersion(row["version_id"], row["session_id"], row["seq"], row["content"],
                             citations, row["state"], row["created_by"], row["created_at"],
                             row["decided_by"], row["decided_at"], row["decision_reason"])
