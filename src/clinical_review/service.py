"""问答复核应用服务：提交、审核、发布、升级、撤回与恢复。

- 时钟通过 clock 注入，验收测试可控制时间。
- 所有写操作在单连接锁内完成并即时提交；并发审核由状态条件更新守门。
- 提交与回调均支持幂等键，重复提交/延迟回调不会绕过确认门槛。
"""
from __future__ import annotations

from typing import Callable

from .domain import (
    AnswerVersion, Citation, ConflictError, Evidence, FINAL_SESSION_STATES,
    PermissionError_, ReviewError, Session, SessionState, VersionState,
    check_publishable, decide_route, utcnow_iso,
)
from .store import Store

ROLES = {"reviewer", "doctor"}


class Service:
    def __init__(self, store: Store | None = None, clock: Callable[[], str] = utcnow_iso) -> None:
        self.store = store or Store()
        self.clock = clock

    # ---- 基础 ----
    def health(self) -> dict[str, str]:
        return {"service": "clinical_review", "status": "ok"}

    def _now(self) -> str:
        return self.clock()

    def _check_role(self, role: str) -> None:
        if role not in ROLES:
            raise PermissionError_(f"未知角色: {role}")

    def _idempotent(self, key: str | None, scope: str, fn):
        """有幂等键时：已见则返回首次结果；否则执行并记录。锁内完成，防并发重复。"""
        if not key:
            return fn()
        with self.store.lock:
            hit = self.store.get_idempotent(key)
            if hit is not None:
                hit = dict(hit, idempotent_replay=True)
                return hit
            result = fn()
            self.store.put_idempotent(key, scope, result, self._now())
            self.store.commit()
            return result

    # ---- 证据集 ----
    def register_evidence(self, evidence_id: str, title: str, valid_until: str) -> dict:
        with self.store.lock:
            self.store.upsert_evidence(Evidence(evidence_id, title, valid_until), self._now())
            self.store.commit()
        return {"evidence_id": evidence_id, "status": "active"}

    def supersede_evidence(self, evidence_id: str, actor: str, role: str, reason: str = "") -> dict:
        """新证据否定旧证据：作废证据，撤回相关已发布版本，通知仍在处理的会话。"""
        self._check_role(role)
        if role != "doctor":
            raise PermissionError_("否定证据并撤回已发布内容必须由医生执行")
        now = self._now()
        with self.store.lock:
            ev = self.store.get_evidence(evidence_id)
            if ev is None:
                raise ReviewError(f"证据不存在: {evidence_id}")
            self.store.upsert_evidence(
                Evidence(ev.evidence_id, ev.title, ev.valid_until, "superseded"), now)

            withdrawn, notified = [], set()
            for v in self.store.list_published_citing(evidence_id):
                self.store.set_version_state(v.version_id, VersionState.WITHDRAWN.value, now,
                                             reason or "引用依据被新证据否定")
                self.store.update_session_state(v.session_id, SessionState.OPEN.value, now)
                self.store.add_notification(
                    v.session_id, "version_withdrawn",
                    f"已发布版本 {v.version_id} 因证据 {evidence_id} 被否定而撤回，需重新审核", now)
                self.store.add_event(v.session_id, v.version_id, actor, role,
                                     "withdraw", reason or "证据被否定", now)
                withdrawn.append(v.version_id)
                notified.add(v.session_id)
            for v in self.store.list_pending_citing(evidence_id):
                if v.session_id not in notified:
                    self.store.add_notification(
                        v.session_id, "evidence_superseded",
                        f"待审核版本 {v.version_id} 引用的证据 {evidence_id} 已被否定，请更新引用", now)
                    notified.add(v.session_id)
            self.store.commit()
        return {"evidence_id": evidence_id, "status": "superseded",
                "withdrawn_versions": withdrawn, "notified_sessions": sorted(notified)}

    # ---- 会话与提交 ----
    def create_session(self, session_id: str, question: str,
                       risk_labels: list[str], idempotency_key: str | None = None) -> dict:
        def _create() -> dict:
            now = self._now()
            with self.store.lock:
                if self.store.get_session(session_id) is not None:
                    raise ConflictError(f"会话已存在: {session_id}")
                s = Session(session_id, question, tuple(risk_labels),
                            SessionState.OPEN.value, now, now)
                self.store.insert_session(s)
                self.store.commit()
            return {"session_id": session_id, "state": s.state,
                    "route": decide_route(s.risk_labels).route}
        return self._idempotent(idempotency_key, "create_session", _create)

    def submit_answer(self, session_id: str, version_id: str, content: str,
                      citations: list[dict], actor: str, role: str = "reviewer",
                      idempotency_key: str | None = None) -> dict:
        """提交回答版本。急症会话在此直接转人工，不进入待审核队列。"""
        self._check_role(role)

        def _submit() -> dict:
            now = self._now()
            with self.store.lock:
                session = self.store.get_session(session_id)
                if session is None:
                    raise ReviewError(f"会话不存在: {session_id}")
                if session.state != SessionState.OPEN.value:
                    raise ConflictError(f"会话状态为 {session.state}，不能提交新回答")
                route = decide_route(session.risk_labels)
                state = (VersionState.ESCALATED if route.route == "escalate"
                         else VersionState.PENDING).value
                v = AnswerVersion(
                    version_id=version_id, session_id=session_id,
                    seq=self.store.next_seq(session_id), content=content,
                    citations=tuple(Citation(**c) for c in citations), state=state,
                    created_by=actor, created_at=now,
                    decided_by=actor if route.route == "escalate" else "",
                    decided_at=now if route.route == "escalate" else "",
                    decision_reason="; ".join(route.reasons))
                self.store.insert_version(v)
                self.store.add_event(session_id, version_id, actor, role, "submit",
                                     f"route={route.route}", now)
                if route.route == "escalate":
                    self.store.update_session_state(session_id, SessionState.ESCALATED.value, now)
                    self.store.add_event(session_id, version_id, actor, role, "escalate",
                                         "; ".join(route.reasons), now)
                    self.store.add_notification(session_id, "escalated",
                                                "命中急症风险，已直接转人工处理", now)
                self.store.commit()
            return {"version_id": version_id, "session_id": session_id, "state": state,
                    "route": route.route, "reasons": list(route.reasons)}
        return self._idempotent(idempotency_key, "submit_answer", _submit)

    # ---- 审核动作 ----
    def propose_edit(self, version_id: str, new_version_id: str, content: str,
                     citations: list[dict], actor: str, role: str) -> dict:
        """审核者提出修改：原版本被取代，生成新的待审核版本。"""
        self._check_role(role)
        now = self._now()
        with self.store.lock:
            old = self.store.get_version(version_id)
            if old is None:
                raise ReviewError(f"版本不存在: {version_id}")
            if old.state != VersionState.PENDING.value:
                raise ConflictError(f"版本状态为 {old.state}，不能修改")
            if self.store.decide_version(version_id, VersionState.SUPERSEDED.value,
                                         actor, now, "被修改版取代") != 1:
                raise ConflictError("版本已被其他审核者处理")
            v = AnswerVersion(
                version_id=new_version_id, session_id=old.session_id,
                seq=self.store.next_seq(old.session_id), content=content,
                citations=tuple(Citation(**c) for c in citations),
                created_by=actor, created_at=now)
            self.store.insert_version(v)
            self.store.add_event(old.session_id, new_version_id, actor, role,
                                 "propose_edit", f"取代 {version_id}", now)
            self.store.commit()
        return {"version_id": new_version_id, "session_id": old.session_id,
                "state": v.state, "supersedes": version_id}

    def return_version(self, version_id: str, actor: str, role: str, reason: str = "") -> dict:
        """退回：版本打回给提交方，会话保持 open 等待新回答。"""
        self._check_role(role)
        return self._decide(version_id, VersionState.RETURNED, actor, role, "return", reason)

    def escalate(self, version_id: str, actor: str, role: str, reason: str = "") -> dict:
        """人工升级：审核中发现风险，转人工处理。"""
        self._check_role(role)
        result = self._decide(version_id, VersionState.ESCALATED, actor, role, "escalate", reason)
        now = self._now()
        with self.store.lock:
            self.store.update_session_state(result["session_id"], SessionState.ESCALATED.value, now)
            self.store.add_notification(result["session_id"], "escalated",
                                        f"版本 {version_id} 被升级转人工: {reason}", now)
            self.store.commit()
        return result

    def publish(self, version_id: str, actor: str, role: str) -> dict:
        """发布：通过全部门槛校验后生效；确认门槛不可被重复提交或延迟回调绕过。"""
        self._check_role(role)
        now = self._now()
        with self.store.lock:
            v = self.store.get_version(version_id)
            if v is None:
                raise ReviewError(f"版本不存在: {version_id}")
            session = self.store.get_session(v.session_id)
            evidence = self.store.evidence_map([c.evidence_id for c in v.citations])
            check_publishable(session, v, evidence, now, role)
            if self.store.decide_version(version_id, VersionState.PUBLISHED.value,
                                         actor, now) != 1:
                raise ConflictError("版本已被其他审核者处理")
            self.store.update_session_state(v.session_id, SessionState.ANSWERED.value, now)
            self.store.add_event(v.session_id, version_id, actor, role, "publish", "", now)
            self.store.commit()
        return {"version_id": version_id, "session_id": v.session_id,
                "state": VersionState.PUBLISHED.value}

    def _decide(self, version_id: str, target: VersionState, actor: str,
                role: str, action: str, reason: str) -> dict:
        now = self._now()
        with self.store.lock:
            v = self.store.get_version(version_id)
            if v is None:
                raise ReviewError(f"版本不存在: {version_id}")
            if self.store.decide_version(version_id, target.value, actor, now, reason) != 1:
                raise ConflictError(f"版本状态为 {v.state}，不能执行 {action}")
            self.store.add_event(v.session_id, version_id, actor, role, action, reason, now)
            self.store.commit()
        return {"version_id": version_id, "session_id": v.session_id, "state": target.value}

    # ---- 延迟回调 ----
    def deliver_callback(self, session_id: str, callback_id: str,
                         action: str = "model_update") -> dict:
        """模型异步回调。同一 callback_id 只生效一次；会话已到终态的延迟回调被忽略。"""
        now = self._now()
        with self.store.lock:
            session = self.store.get_session(session_id)
            if session is None:
                raise ReviewError(f"会话不存在: {session_id}")
            if not self.store.record_callback(callback_id, session_id, False, now):
                self.store.commit()
                return {"callback_id": callback_id, "applied": False, "reason": "重复回调"}
            if session.state in {s.value for s in FINAL_SESSION_STATES}:
                self.store.add_event(session_id, "", "system", "system", "callback_ignored",
                                     f"会话已终态({session.state})，忽略延迟回调 {callback_id}", now)
                self.store.commit()
                return {"callback_id": callback_id, "applied": False,
                        "reason": f"会话已终态: {session.state}"}
            self.store.mark_callback_applied(callback_id)
            self.store.add_event(session_id, "", "system", "system", "callback_applied",
                                 f"{action}:{callback_id}", now)
            self.store.commit()
        return {"callback_id": callback_id, "applied": True}

    # ---- 查询与恢复 ----
    def get_session(self, session_id: str) -> dict | None:
        s = self.store.get_session(session_id)
        return s.__dict__.copy() if s else None

    def list_versions(self, session_id: str) -> list[dict]:
        return [{**v.__dict__, "citations": [c.__dict__ for c in v.citations]}
                for v in self.store.list_versions(session_id)]

    def list_unfinished(self) -> list[dict]:
        """重启后恢复：所有未完成的会话（待处理或已升级待人工）。"""
        with self.store.lock:
            sessions = self.store.list_sessions_by_state(
                (SessionState.OPEN.value, SessionState.ESCALATED.value))
            return [{**s.__dict__,
                     "pending_versions": [v.version_id for v in
                                          self.store.list_versions(s.session_id)
                                          if v.state == VersionState.PENDING.value]}
                    for s in sessions]

    def list_notifications(self, session_id: str) -> list[dict]:
        return self.store.list_notifications(session_id)

    def list_events(self, session_id: str) -> list[dict]:
        return self.store.list_events(session_id)
