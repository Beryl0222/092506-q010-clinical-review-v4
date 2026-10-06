"""问答复核应用服务。

关键不变量：

1. 急症在提问进入时立即升级人工，模型回答（含延迟/重复回调）不能逆转。
2. 高风险版本必须有医生确认；引用依据在确认时刻必须仍有效，
   重复提交与重复回调都会被指纹/回调 CAS 拦下，不能绕过确认门槛。
3. 所有状态推进都是带期望状态的 CAS，并发审核只会有一个胜出者。
4. 已发布版本被新证据否定时撤回版本，并向仍 active 的会话发通知。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from .domain import (
    CITATION_OK,
    ROLE_DOCTOR,
    ROLE_SYSTEM,
    V_CHANGES,
    V_DRAFT,
    V_ESCALATED,
    V_PENDING,
    V_PUBLISHED,
    V_SUPERSEDED,
    V_WITHDRAWN,
    CASE_AWAITING,
    CASE_CHANGES,
    CASE_ESCALATED,
    CASE_PENDING,
    CASE_PUBLISHED,
    CASE_WITHDRAWN,
    TAG_OUTDATED_CITATION,
    AnswerVersion,
    CallbackRecord,
    Case,
    ConflictError,
    Evidence,
    EvidenceCatalog,
    Event,
    NotFoundError,
    Notification,
    PermissionDeniedError,
    Record,
    RiskRuleEngine,
    SessionRecord,
    StaleCitationError,
    check_citation,
    dedup_fingerprint,
    sanitize_question,
    ts,
)
from .store import Store


def _new_id() -> str:
    return uuid.uuid4().hex


class Service:
    def __init__(self, store: Store | None = None,
                 clock=None,
                 catalog: EvidenceCatalog | None = None,
                 rules: RiskRuleEngine | None = None) -> None:
        self.store = store or Store()
        self.clock = clock
        if self.clock is None:
            from .domain import SystemClock
            self.clock = SystemClock()
        self.catalog = catalog or EvidenceCatalog()
        # 传入的证据集落库（已存在则更新），保证重启后可恢复
        for evidence in self.catalog.all():
            self.store.upsert_evidence(evidence)
        # 再从库中恢复全部证据（含历史上的取代关系）
        for evidence in self.store.list_evidences():
            self.catalog.add(evidence)
        self.rules = rules or RiskRuleEngine(self.catalog)

    @property
    def now(self):
        return self.clock.now()

    # ------------------------------------------------------------------
    # 基线能力
    # ------------------------------------------------------------------

    def health(self) -> dict[str, str]:
        return {"service": "clinical_review", "status": "ok"}

    def register(self, record_id: str, owner_id: str) -> dict[str, str]:
        record = self.store.save(Record(record_id, owner_id))
        return {"record_id": record.record_id, "owner_id": record.owner_id,
                "state": record.state, "created_at": record.created_at}

    def find(self, record_id: str) -> dict[str, str] | None:
        record = self.store.get(record_id)
        return record.__dict__.copy() if record else None

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _get_case(self, case_id: str) -> Case:
        case = self.store.get_case(case_id)
        if case is None:
            raise NotFoundError(f"案件不存在: {case_id}")
        return case

    def _event(self, case_id: str, kind: str, actor: str,
               payload: dict[str, Any] | None = None,
               version_id: str | None = None) -> None:
        self.store.insert_event(Event(
            event_id=_new_id(), case_id=case_id, version_id=version_id,
            kind=kind, actor=actor,
            payload=json.dumps(payload or {}, ensure_ascii=False),
            created_at=ts(self.now),
        ))

    def _notify(self, case: Case, kind: str, payload: dict[str, Any],
                version_id: str | None = None) -> None:
        """只通知仍在处理（active）的会话。"""
        if not case.session_id:
            return
        session = self.store.get_session(case.session_id)
        if session is None or session.status != "active":
            return
        self.store.insert_notification(Notification(
            notification_id=_new_id(), session_id=case.session_id,
            case_id=case.case_id, version_id=version_id, kind=kind,
            payload=json.dumps(payload, ensure_ascii=False),
            created_at=ts(self.now),
        ))

    def _citation_bad(self, citations: tuple[str, ...]) -> dict[str, str]:
        return {cid: state for cid, state in (
            (cid, check_citation(self.catalog.get(cid), self.now))
            for cid in citations
        ) if state != CITATION_OK}

    # ------------------------------------------------------------------
    # 提问接入 / 脱敏 / 去重 / 急症升级
    # ------------------------------------------------------------------

    def open_session(self, session_id: str) -> dict:
        existing = self.store.get_session(session_id)
        if existing and existing.status == "active":
            return self._session_dict(existing)
        record = SessionRecord(session_id=session_id, status="active",
                               created_at=ts(self.now))
        self.store.upsert_session(record)
        return self._session_dict(record)

    def close_session(self, session_id: str) -> dict:
        session = self.store.get_session(session_id)
        if session is None:
            raise NotFoundError(f"会话不存在: {session_id}")
        record = SessionRecord(session_id=session_id, status="closed",
                               created_at=session.created_at,
                               closed_at=ts(self.now))
        self.store.upsert_session(record)
        return self._session_dict(record)

    def submit_question(self, case_id: str, question: str,
                        session_id: str | None = None,
                        actor: str = ROLE_SYSTEM) -> dict:
        """登记脱敏提问；急症立即转人工，其余进入等待模型回答状态。

        相同脱敏提问的重复提交直接返回既有案件，不会新建绕过门槛的流程。
        """
        question_safe = sanitize_question(question)
        fingerprint = dedup_fingerprint(question_safe)
        now = ts(self.now)

        with self.store.transaction():
            if session_id:
                self.open_session(session_id)
            existing = self.store.get_case_by_dedup(fingerprint)
            if existing is not None:
                result = self._case_dict(existing)
                result["duplicate"] = True
                return result

            assessment = self.rules.evaluate(question_safe, None, None, self.now)
            emergency = "RED_EMERGENCY" in assessment.tags
            status = CASE_ESCALATED if emergency else CASE_AWAITING

            case = Case(
                case_id=case_id, dedup_key=fingerprint,
                question_safe=question_safe, status=status,
                session_id=session_id,
                require_doctor=False, stale=False,
                created_at=now, updated_at=now,
            )
            try:
                self.store.insert_case(case)
            except sqlite3.IntegrityError:
                # 并发下唯一索引竞争：以先到的案件为准
                existing = self.store.get_case_by_dedup(fingerprint)
                if existing is not None:
                    result = self._case_dict(existing)
                    result["duplicate"] = True
                    return result
                raise

            if emergency:
                # 不产生任何待自动回复的版本，直接人工
                self._event(case_id, "escalated_human", actor,
                            {"reasons": list(assessment.reasons),
                             "tags": list(assessment.tags)})
                self._notify(case, "escalated_human",
                             {"reason": "疑似急症，已直接转人工"})
                return self._case_dict(self._get_case(case_id)) | {
                    "duplicate": False}

            version = AnswerVersion(
                version_id=f"{case_id}:v1", case_id=case_id, seq=1,
                answer_text="", citations=(), risk_tags=assessment.tags,
                route=assessment.route, require_doctor=False,
                status=V_DRAFT, reasons=assessment.reasons,
                created_by=actor, created_at=now,
            )
            self.store.insert_version(version)
            self.store.cas_case(case_id, (CASE_AWAITING,),
                                {"current_version_id": version.version_id,
                                 "updated_at": now})
            self._event(case_id, "question_submitted", actor,
                        {"session_id": session_id}, version.version_id)
            result = self._case_dict(self._get_case(case_id))
            result["duplicate"] = False
            return result

    def register_callback(self, callback_id: str, case_id: str) -> dict:
        """模型回调登记，幂等。急症案件不接受任何自动回复回调。"""
        with self.store.transaction():
            case = self._get_case(case_id)
            if case.status == CASE_ESCALATED:
                raise ConflictError("急症案件已转人工，不接受自动回复回调")
            if case.current_version_id is None:
                raise ConflictError("该案件没有等待回答的版本")
            record = CallbackRecord(callback_id=callback_id, case_id=case_id,
                                    version_id=case.current_version_id,
                                    status="pending", created_at=ts(self.now))
            saved, created = self.store.insert_callback_if_absent(record)
            return {"callback_id": saved.callback_id, "case_id": saved.case_id,
                    "version_id": saved.version_id, "status": saved.status,
                    "duplicate": not created}

    def deliver_answer(self, callback_id: str, answer_text: str,
                       citations: list[str] | None = None) -> dict:
        """模型延迟回调送达。重复/迟到回调不能改变已定稿的版本。"""
        with self.store.transaction():
            callback = self.store.get_callback(callback_id)
            if callback is None:
                raise NotFoundError(f"回调未登记: {callback_id}")

            case = self._get_case(callback.case_id)
            version = self.store.get_version(callback.version_id)
            if version is None:
                raise NotFoundError("回调对应的版本不存在")

            # 回调 CAS：只有第一个 pending -> done 生效
            if not self.store.cas_callback(callback_id, ts(self.now)):
                result = self._case_dict(case)
                result["duplicate"] = True
                result["version"] = self._version_dict(version)
                return result

            citations = tuple(citations or [])

            # 急症案件：自动回答一律忽略，保持人工升级状态；
            # 已离开等待状态的案件同样不接受迟到回答
            if case.status != CASE_AWAITING:
                self._event(case.case_id, "late_answer_ignored", ROLE_SYSTEM,
                            {"callback_id": callback_id,
                             "case_status": case.status},
                            version.version_id)
                result = self._case_dict(case)
                result["duplicate"] = False
                result["ignored"] = True
                return result

            assessment = self.rules.evaluate(
                case.question_safe, answer_text, list(citations), self.now)
            now = ts(self.now)

            if "RED_EMERGENCY" in assessment.tags:
                # 回答中暴露急症：立即升级，不等待任何确认
                self.store.cas_version(version.version_id, (V_DRAFT,),
                                       {"status": V_ESCALATED,
                                        "answer_text": answer_text,
                                        "citations": json.dumps(list(citations)),
                                        "risk_tags": json.dumps(
                                            list(assessment.tags)),
                                        "decided_at": now})
                self.store.cas_case(case.case_id, (CASE_AWAITING,),
                                    {"status": CASE_ESCALATED,
                                     "updated_at": now})
                self._event(case.case_id, "escalated_human", ROLE_SYSTEM,
                            {"reasons": list(assessment.reasons)},
                            version.version_id)
                self._notify(case, "escalated_human",
                             {"reason": "回答中出现急症线索，已转人工"},
                             version.version_id)
                return self._case_dict(self._get_case(case.case_id)) | {
                    "duplicate": False}

            self.store.cas_version(version.version_id, (V_DRAFT,), {
                "answer_text": answer_text,
                "citations": json.dumps(list(citations)),
                "risk_tags": json.dumps(list(assessment.tags)),
                "route": assessment.route,
                "require_doctor": int(assessment.require_doctor),
                "status": V_PENDING,
                "reasons": json.dumps(list(assessment.reasons),
                                      ensure_ascii=False),
                "decided_at": now,
            })
            self.store.cas_case(case.case_id, (CASE_AWAITING,), {
                "status": CASE_PENDING,
                "require_doctor": int(assessment.require_doctor),
                "stale": int(TAG_OUTDATED_CITATION in assessment.tags),
                "updated_at": now,
            })
            self._event(case.case_id, "answer_received", ROLE_SYSTEM,
                        {"tags": list(assessment.tags),
                         "citations": list(citations),
                         "require_doctor": assessment.require_doctor},
                        version.version_id)
            result = self._case_dict(self._get_case(case.case_id))
            result["duplicate"] = False
            result["version"] = self._version_dict(
                self.store.get_version(version.version_id))
            return result

    # ------------------------------------------------------------------
    # 审核队列 / 认领 / 权限
    # ------------------------------------------------------------------

    def list_queue(self, statuses: tuple[str, ...] | None = None) -> list[dict]:
        return [self._case_dict(case) for case in self.store.list_cases(statuses)]

    def claim(self, case_id: str, actor_id: str, role: str) -> dict:
        if role not in ("reviewer", "doctor"):
            raise PermissionDeniedError("只有审核者或医生可以认领案件")
        with self.store.transaction():
            case = self._get_case(case_id)
            if case.claimed_by == actor_id:
                return self._case_dict(case)
            if case.status != CASE_PENDING:
                raise ConflictError(f"案件当前状态不可认领: {case.status}")
            # 高风险案件审核者也可以先做初审（退回/预确认），
            # 但发布门槛仍要求医生确认，规则在 confirm/publish 处强制
            if not self.store.cas_case(
                    case_id, (CASE_PENDING,),
                    {"claimed_by": actor_id, "updated_at": ts(self.now)},
                    expected_unclaimed=True):
                raise ConflictError("案件已被其他审核者认领")
            self._event(case_id, "claimed", actor_id, {"role": role})
            return self._case_dict(self._get_case(case_id))

    def release_claim(self, case_id: str, actor_id: str) -> dict:
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            self.store.cas_case(case_id, (CASE_PENDING, CASE_CHANGES),
                                {"claimed_by": None, "updated_at": ts(self.now)})
            self._event(case_id, "claim_released", actor_id)
            return self._case_dict(self._get_case(case_id))

    def _require_claim(self, case_id: str, actor_id: str) -> Case:
        case = self._get_case(case_id)
        if case.claimed_by != actor_id:
            raise PermissionDeniedError("只有认领该案件的审核者可以操作")
        return case

    def _pending_version(self, case: Case) -> AnswerVersion:
        if not case.current_version_id:
            raise ConflictError("案件没有可处理的版本")
        version = self.store.get_version(case.current_version_id)
        if version is None or version.status != V_PENDING:
            raise ConflictError(f"版本当前状态不可处理: "
                                f"{version.status if version else None}")
        return version

    # ------------------------------------------------------------------
    # 退回 / 修改
    # ------------------------------------------------------------------

    def request_changes(self, case_id: str, actor_id: str, role: str,
                        comment: str = "") -> dict:
        if role not in ("reviewer", "doctor"):
            raise PermissionDeniedError("无退回权限")
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            version = self._pending_version(case)
            now = ts(self.now)
            if not self.store.cas_version(version.version_id, (V_PENDING,),
                                          {"status": V_CHANGES,
                                           "decided_at": now}):
                raise ConflictError("版本状态已变化，退回失败")
            self.store.cas_case(case_id, (CASE_PENDING,),
                                {"status": CASE_CHANGES, "stale": 0,
                                 "updated_at": now})
            self.store.add_confirmation(version.version_id, case_id, actor_id,
                                        "returned", now, comment=comment)
            self._event(case_id, "changes_requested", actor_id,
                        {"comment": comment}, version.version_id)
            return self._case_dict(self._get_case(case_id))

    def reopen_for_revision(self, case_id: str, actor_id: str, role: str,
                            comment: str = "") -> dict:
        """对已发布/已撤回内容提出修改并认领，随后可调用 revise。"""
        if role not in ("reviewer", "doctor"):
            raise PermissionDeniedError("无提出修改权限")
        with self.store.transaction():
            case = self._get_case(case_id)
            if case.claimed_by is not None and case.claimed_by != actor_id:
                raise ConflictError("案件正被其他审核者处理")
            if case.status not in (CASE_PUBLISHED, CASE_WITHDRAWN) \
                    or not case.current_version_id:
                raise ConflictError("只有已发布或已撤回案件可以提出修改")
            now = ts(self.now)
            self.store.cas_case(case_id, (CASE_PUBLISHED, CASE_WITHDRAWN),
                                {"status": CASE_CHANGES,
                                 "claimed_by": actor_id,
                                 "updated_at": now})
            self._event(case_id, "revision_requested_published", actor_id,
                        {"comment": comment}, case.current_version_id)
            return self._case_dict(self._get_case(case_id))

    def revise(self, case_id: str, actor_id: str, answer_text: str,
               citations: list[str] | None = None, comment: str = "") -> dict:
        """根据修改意见产生新版本，重新走规则与确认门槛。"""
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            if case.status != CASE_CHANGES:
                raise ConflictError(f"只有退回状态可以修改: {case.status}")
            old = self.store.get_version(case.current_version_id)
            if old is None or old.status not in (
                    V_CHANGES, V_PUBLISHED, V_WITHDRAWN):
                raise ConflictError("上一版本状态异常")

            citations_t = tuple(citations or [])
            assessment = self.rules.evaluate(
                case.question_safe, answer_text, list(citations_t), self.now)
            now = ts(self.now)
            seq = self.store.next_seq(case_id)
            new_version = AnswerVersion(
                version_id=f"{case_id}:v{seq}", case_id=case_id, seq=seq,
                answer_text=answer_text, citations=citations_t,
                risk_tags=assessment.tags, route=assessment.route,
                require_doctor=assessment.require_doctor,
                status=V_PENDING, reasons=assessment.reasons,
                created_by=actor_id, created_at=now, decided_at=now,
            )
            self.store.insert_version(new_version)
            # 被退回的版本被取代；仍在发布/已撤回的版本保持原状态，
            # 待新版本发布时再由 _publish 取代旧发布
            if old.status == V_CHANGES:
                self.store.cas_version(old.version_id, (V_CHANGES,),
                                       {"status": V_SUPERSEDED})
            if "RED_EMERGENCY" in assessment.tags:
                self.store.cas_case(case_id, (CASE_CHANGES,),
                                    {"status": CASE_ESCALATED,
                                     "current_version_id": new_version.version_id,
                                     "require_doctor": 0, "stale": 0,
                                     "claimed_by": None,
                                     "updated_at": now})
                self.store.cas_version(new_version.version_id, (V_PENDING,),
                                       {"status": V_ESCALATED})
                self._event(case_id, "escalated_human", actor_id,
                            {"reasons": list(assessment.reasons)},
                            new_version.version_id)
                self._notify(case, "escalated_human",
                             {"reason": "修改稿出现急症线索，已转人工"},
                             new_version.version_id)
                return self._case_dict(self._get_case(case_id))

            self.store.cas_case(case_id, (CASE_CHANGES,), {
                "status": CASE_PENDING,
                "current_version_id": new_version.version_id,
                "require_doctor": int(assessment.require_doctor),
                "stale": int(TAG_OUTDATED_CITATION in assessment.tags),
                # 修订稿重新入队；若升级为高风险，审核者继续占着认领
                # 会挡住医生，故一律释放
                "claimed_by": None,
                "updated_at": now,
            })
            self._event(case_id, "revised", actor_id,
                        {"comment": comment, "old_version": old.version_id,
                         "tags": list(assessment.tags)},
                        new_version.version_id)
            return self._case_dict(self._get_case(case_id))

    # ------------------------------------------------------------------
    # 确认 / 发布
    # ------------------------------------------------------------------

    def _enforce_fresh(self, version_id: str, actor_role: str) -> AnswerVersion:
        """在独立事务中核对引用；失效则打标提交后抛异常。

        异常在事务正常提交之后才向外抛，避免外层回滚把
        “升级医生门槛”的更新一起撤销。审核者触发时释放认领，
        让医生可以接手；医生本人触发时保留认领，方便其退回/修改。
        """
        stale: StaleCitationError | None = None
        with self.store.transaction():
            version = self.store.get_version(version_id)
            bad = self._citation_bad(version.citations)
            if bad:
                tags = sorted(set(version.risk_tags) | {TAG_OUTDATED_CITATION})
                now = ts(self.now)
                self.store.cas_version(version.version_id, (V_PENDING,), {
                    "risk_tags": json.dumps(tags, ensure_ascii=False),
                    "require_doctor": 1,
                    "route": "doctor_confirmation",
                })
                changes = {"require_doctor": 1, "stale": 1,
                           "updated_at": now}
                if actor_role != ROLE_DOCTOR:
                    changes["claimed_by"] = None
                self.store.cas_case(version.case_id, (CASE_PENDING,), changes)
                self._event(version.case_id, "citation_expired", ROLE_SYSTEM,
                            {"bad": bad, "version_id": version.version_id},
                            version.version_id)
                stale = StaleCitationError(f"引用依据已失效: {bad}")
        if stale is not None:
            raise stale
        return self.store.get_version(version_id)

    def confirm(self, case_id: str, actor_id: str, role: str,
                comment: str = "") -> dict:
        if role not in ("reviewer", "doctor"):
            raise PermissionDeniedError("无确认权限")
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            version = self._pending_version(case)
            version_id = version.version_id
        # 门槛在确认时刻重新验证引用，过期则打标并阻断（独立提交）
        self._enforce_fresh(version_id, role)
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            version = self._pending_version(case)
            if self._citation_bad(version.citations):
                raise StaleCitationError("引用依据已失效，确认被阻断")
            now = ts(self.now)

            if role == ROLE_DOCTOR:
                self.store.add_confirmation(version.version_id, case_id,
                                            actor_id, "doctor", now,
                                            doctor_id=actor_id, comment=comment)
                self._event(case_id, "doctor_confirmed", actor_id,
                            {"comment": comment}, version.version_id)
                return self._publish(case, version, actor_id)

            # reviewer 确认
            if version.require_doctor:
                # 记录审核意见，但不满足医生门槛，不得发布；
                # 释放认领，让医生可以从队列中认领
                self.store.add_confirmation(version.version_id, case_id,
                                            actor_id, "reviewer", now,
                                            reviewer_id=actor_id,
                                            comment=comment)
                self.store.cas_case(case_id, (CASE_PENDING,),
                                    {"claimed_by": None,
                                     "updated_at": now})
                self._event(case_id, "reviewer_confirmed_pending_doctor",
                            actor_id, {"comment": comment}, version.version_id)
                result = self._case_dict(self._get_case(case_id))
                result["published"] = False
                result["gate"] = "waiting_doctor"
                return result

            self.store.add_confirmation(version.version_id, case_id,
                                        actor_id, "reviewer", now,
                                        reviewer_id=actor_id, comment=comment)
            self._event(case_id, "reviewer_confirmed", actor_id,
                        {"comment": comment}, version.version_id)
            return self._publish(case, version, actor_id)

    def publish(self, case_id: str, actor_id: str, role: str) -> dict:
        if role not in ("reviewer", "doctor"):
            raise PermissionDeniedError("无发布权限")
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            version = self._pending_version(case)
            version_id = version.version_id
        # 发布前最后一刻仍要确认引用有效（过期会独立提交打标后抛错）
        self._enforce_fresh(version_id, role)
        with self.store.transaction():
            case = self._require_claim(case_id, actor_id)
            version = self._pending_version(case)
            # 门槛校验失败时整个事务回滚，认领保持不变；
            # 审核者可显式 confirm（登记意见并释放）或 release_claim 交接
            if self._citation_bad(version.citations):
                raise StaleCitationError("引用依据已失效，不能发布")
            if version.require_doctor and not self.store.has_confirmation(
                    version_id, "doctor"):
                raise ConflictError("缺少医生确认，不能发布")
            if not (self.store.has_confirmation(version_id, "reviewer")
                    or self.store.has_confirmation(version_id, "doctor")):
                raise ConflictError("缺少审核确认，不能发布")
            self._event(case_id, "published", actor_id, {}, version_id)
            return self._publish(case, version, actor_id, record_event=False)

    def _publish(self, case: Case, version: AnswerVersion, actor_id: str,
                 record_event: bool = True) -> dict:
        now = ts(self.now)
        if not self.store.cas_version(version.version_id, (V_PENDING,),
                                      {"status": V_PUBLISHED,
                                       "decided_at": now}):
            raise ConflictError("版本已被并发处理，发布失败")
        # 同案件的其它发布版本自动被新版本取代
        for other in self.store.list_versions(case.case_id):
            if other.version_id != version.version_id \
                    and other.status == V_PUBLISHED:
                self.store.cas_version(other.version_id, (V_PUBLISHED,),
                                       {"status": V_SUPERSEDED})
        self.store.cas_case(case.case_id, (CASE_PENDING,),
                            {"status": CASE_PUBLISHED, "stale": 0,
                             "claimed_by": None,
                             "updated_at": now})
        if record_event:
            self._event(case.case_id, "published", actor_id,
                        {"version_id": version.version_id}, version.version_id)
        fresh = self._get_case(case.case_id)
        self._notify(fresh, "answer_published",
                     {"version_id": version.version_id}, version.version_id)
        result = self._case_dict(fresh)
        result["published"] = True
        result["version"] = self._version_dict(
            self.store.get_version(version.version_id))
        return result

    # ------------------------------------------------------------------
    # 撤回 / 新证据否定 / 版本回滚
    # ------------------------------------------------------------------

    def _withdraw_version(self, version: AnswerVersion, reason: str,
                          actor: str) -> None:
        """撤回单个已发布版本并通知会话；不假设案件仍处于已发布状态。"""
        now = ts(self.now)
        if not self.store.cas_version(version.version_id, (V_PUBLISHED,),
                                      {"status": V_WITHDRAWN,
                                       "decided_at": now}):
            raise ConflictError("版本状态已变化，撤回失败")
        case = self._get_case(version.case_id)
        if case.status == CASE_PUBLISHED:
            self.store.cas_case(case.case_id, (CASE_PUBLISHED,),
                                {"status": CASE_WITHDRAWN, "stale": 1,
                                 "claimed_by": None,
                                 "updated_at": now})
        else:
            # 案件可能正处于修改流；旧版本下线同样要让会话知晓
            self._event(case.case_id, "published_version_invalidated",
                        ROLE_SYSTEM, {"version_id": version.version_id},
                        version.version_id)
        self._event(case.case_id, "withdrawn", actor,
                    {"reason": reason, "version_id": version.version_id},
                    version.version_id)
        fresh = self._get_case(case.case_id)
        self._notify(fresh, "answer_withdrawn",
                     {"reason": reason,
                      "version_id": version.version_id},
                     version.version_id)

    def withdraw_published(self, case_id: str, reason: str,
                           actor: str = ROLE_SYSTEM,
                           actor_role: str = ROLE_SYSTEM) -> dict:
        with self.store.transaction():
            case = self._get_case(case_id)
            if actor_role not in (ROLE_SYSTEM, "doctor"):
                raise PermissionDeniedError("只有系统或医生可以撤回发布")
            if case.status != CASE_PUBLISHED or not case.current_version_id:
                raise ConflictError("只有已发布案件可以撤回")
            version = self.store.get_version(case.current_version_id)
            self._withdraw_version(version, reason, actor)
            return self._case_dict(self._get_case(case_id))

    def add_evidence(self, evidence: Evidence) -> None:
        self.catalog.add(evidence)
        self.store.upsert_evidence(evidence)

    def supersede_evidence(self, old_id: str, new_id: str,
                           reason: str = "新证据否定") -> dict:
        """登记新证据否定旧证据；波及在审/已发布版本。"""
        if self.catalog.get(old_id) is None:
            raise NotFoundError(f"证据不存在: {old_id}")
        with self.store.transaction():
            self.catalog.supersede(old_id, new_id)
            # 取代关系落库，重启后仍生效
            self.store.upsert_evidence(self.catalog.get(old_id))
            affected_pending: list[str] = []
            withdrawn: list[str] = []

            for version in self.store.list_versions_by_status(
                    (V_PENDING, V_PUBLISHED)):
                if old_id not in version.citations:
                    continue
                case = self._get_case(version.case_id)
                if version.status == V_PUBLISHED:
                    self._withdraw_version(
                        version,
                        f"{reason}: 证据 {old_id} 被 {new_id} 取代",
                        ROLE_SYSTEM)
                    withdrawn.append(case.case_id)
                    continue

                tags = set(version.risk_tags)
                tags.add(TAG_OUTDATED_CITATION)
                self.store.cas_version(version.version_id, (V_PENDING,), {
                    "risk_tags": json.dumps(sorted(tags), ensure_ascii=False),
                    "require_doctor": 1, "route": "doctor_confirmation",
                })
                self.store.cas_case(case.case_id, (CASE_PENDING,),
                                    {"require_doctor": 1, "stale": 1,
                                     "updated_at": ts(self.now)})
                self._event(case.case_id, "evidence_superseded", ROLE_SYSTEM,
                            {"old": old_id, "new": new_id}, version.version_id)
                affected_pending.append(case.case_id)

            return {"old": old_id, "new": new_id,
                    "pending_blocked": affected_pending,
                    "published_withdrawn": withdrawn}

    def sweep_stale_citations(self) -> dict:
        """时间推进后的清扫：在审版本引用过期 -> 升级为医生门槛并打标。"""
        with self.store.transaction():
            blocked: list[str] = []
            for version in self.store.list_versions_by_status((V_PENDING,)):
                bad = self._citation_bad(version.citations)
                if not bad:
                    continue
                tags = set(version.risk_tags)
                tags.add(TAG_OUTDATED_CITATION)
                self.store.cas_version(version.version_id, (V_PENDING,), {
                    "risk_tags": json.dumps(sorted(tags), ensure_ascii=False),
                    "require_doctor": 1, "route": "doctor_confirmation",
                })
                self.store.cas_case(version.case_id, (CASE_PENDING,),
                                    {"require_doctor": 1, "stale": 1,
                                     "updated_at": ts(self.now)})
                self._event(version.case_id, "citation_expired", ROLE_SYSTEM,
                            {"bad": bad}, version.version_id)
                blocked.append(version.case_id)
            return {"blocked": blocked}

    def rollback_to(self, case_id: str, actor_id: str, role: str,
                    target_version_id: str) -> dict:
        """回滚到既往发布版本。仅医生可执行；目标引用当前必须仍有效。"""
        if role != ROLE_DOCTOR:
            raise PermissionDeniedError("只有医生可以回滚已发布版本")
        with self.store.transaction():
            case = self._get_case(case_id)
            # 已发布案件可能仍被原发布者认领；撤回态认领已释放。
            # 医生角色是硬门槛，认领冲突时才拒绝。
            if case.claimed_by is not None and case.claimed_by != actor_id:
                raise PermissionDeniedError("案件正被其他审核者处理")
            if case.status not in (CASE_PUBLISHED, CASE_WITHDRAWN):
                raise ConflictError(f"当前状态不可回滚: {case.status}")
            target = self.store.get_version(target_version_id)
            if target is None or target.case_id != case_id:
                raise NotFoundError("目标版本不存在")
            published_before = any(
                event.kind == "published" and event.version_id == target_version_id
                for event in self.store.list_events(case_id)
            )
            if not published_before:
                raise ConflictError("只能回滚到历史上正式发布过的版本")
            bad = self._citation_bad(target.citations)
            if bad:
                raise StaleCitationError(f"目标版本引用已失效: {bad}")

            current = self.store.get_version(case.current_version_id)
            now = ts(self.now)
            if current and current.version_id != target.version_id:
                if current.status == V_PUBLISHED:
                    self.store.cas_version(current.version_id, (V_PUBLISHED,),
                                           {"status": V_SUPERSEDED})
                self.store.cas_version(target.version_id,
                                       (V_SUPERSEDED, V_WITHDRAWN),
                                       {"status": V_PUBLISHED,
                                        "decided_at": now})
            else:
                self.store.cas_version(target.version_id,
                                       (V_WITHDRAWN, V_SUPERSEDED, V_PUBLISHED),
                                       {"status": V_PUBLISHED,
                                        "decided_at": now})
            self.store.cas_case(case_id, (CASE_PUBLISHED, CASE_WITHDRAWN),
                                {"status": CASE_PUBLISHED, "stale": 0,
                                 "current_version_id": target.version_id,
                                 "updated_at": now})
            self._event(case_id, "rolled_back", actor_id,
                        {"from_version": current.version_id if current else None,
                         "to_version": target.version_id},
                        target.version_id)
            fresh = self._get_case(case_id)
            self._notify(fresh, "answer_rollback",
                         {"version_id": target.version_id},
                         target.version_id)
            result = self._case_dict(fresh)
            result["version"] = self._version_dict(
                self.store.get_version(target.version_id))
            return result

    # ------------------------------------------------------------------
    # 会话通知与重启恢复
    # ------------------------------------------------------------------

    def drain_notifications(self, session_id: str) -> list[dict]:
        with self.store.transaction():
            items = self.store.list_undelivered(session_id)
            result = []
            for item in items:
                if self.store.mark_delivered(item.notification_id,
                                             ts(self.now)):
                    result.append({
                        "notification_id": item.notification_id,
                        "session_id": item.session_id,
                        "case_id": item.case_id,
                        "version_id": item.version_id,
                        "kind": item.kind,
                        "payload": json.loads(item.payload),
                        "created_at": item.created_at,
                    })
            return result

    def pending_notifications(self, session_id: str) -> list[dict]:
        return [{
            "notification_id": item.notification_id,
            "session_id": item.session_id,
            "case_id": item.case_id,
            "version_id": item.version_id,
            "kind": item.kind,
            "payload": json.loads(item.payload),
            "created_at": item.created_at,
        } for item in self.store.list_undelivered(session_id)]

    def recover(self) -> dict:
        """重启后恢复：列出所有未完成工作与未投递通知。"""
        awaiting = [
            {"callback_id": c.callback_id, "case_id": c.case_id,
             "version_id": c.version_id, "created_at": c.created_at}
            for c in self.store.list_pending_callbacks()
        ]
        pending = [self._case_dict(c) for c in self.store.list_cases(
            (CASE_PENDING, CASE_CHANGES))]
        escalated = [self._case_dict(c) for c in self.store.list_cases(
            (CASE_ESCALATED,))]
        sessions = [self._session_dict(s)
                    for s in self.store.list_active_sessions()]
        notifications = [{
            "notification_id": n.notification_id,
            "session_id": n.session_id,
            "case_id": n.case_id,
            "kind": n.kind,
        } for n in self.store.list_undelivered()]
        return {"awaiting_answer": awaiting, "pending_review": pending,
                "escalated_human": escalated, "active_sessions": sessions,
                "undelivered_notifications": notifications}

    def case_detail(self, case_id: str) -> dict:
        with self.store.transaction():
            case = self._get_case(case_id)
            data = self._case_dict(case)
            data["versions"] = [self._version_dict(v)
                                for v in self.store.list_versions(case_id)]
            data["events"] = [{
                "event_id": e.event_id, "kind": e.kind, "actor": e.actor,
                "payload": json.loads(e.payload),
                "created_at": e.created_at,
            } for e in self.store.list_events(case_id)]
            return data

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    @staticmethod
    def _case_dict(case: Case) -> dict:
        return {
            "case_id": case.case_id,
            "question_safe": case.question_safe,
            "status": case.status,
            "session_id": case.session_id,
            "current_version_id": case.current_version_id,
            "require_doctor": case.require_doctor,
            "claimed_by": case.claimed_by,
            "stale": case.stale,
            "created_at": case.created_at,
            "updated_at": case.updated_at,
        }

    @staticmethod
    def _version_dict(version: AnswerVersion) -> dict:
        return {
            "version_id": version.version_id,
            "case_id": version.case_id,
            "seq": version.seq,
            "answer_text": version.answer_text,
            "citations": list(version.citations),
            "risk_tags": list(version.risk_tags),
            "route": version.route,
            "require_doctor": version.require_doctor,
            "status": version.status,
            "reasons": list(version.reasons),
            "created_by": version.created_by,
            "created_at": version.created_at,
            "decided_at": version.decided_at,
        }

    @staticmethod
    def _session_dict(session: SessionRecord) -> dict:
        return {"session_id": session.session_id, "status": session.status,
                "created_at": session.created_at,
                "closed_at": session.closed_at}
