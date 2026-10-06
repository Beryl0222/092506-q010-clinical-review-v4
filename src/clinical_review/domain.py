"""问答复核领域对象：会话、回答版本、证据、引用与风险规则。

状态机：
- 会话：open -> answered / escalated -> closed；已发布版本被撤回时回到 open。
- 版本：pending_review -> published / returned / escalated；
  published -> withdrawn（证据失效）；pending_review -> superseded（被修改版取代）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class Risk(str, Enum):
    EMERGENCY = "emergency"  # 急症：直接转人工，禁止自动回复
    HIGH = "high"            # 高风险：发布前必须由医生确认
    NORMAL = "normal"


class SessionState(str, Enum):
    OPEN = "open"
    ANSWERED = "answered"
    ESCALATED = "escalated"
    CLOSED = "closed"


class VersionState(str, Enum):
    PENDING = "pending_review"
    PUBLISHED = "published"
    RETURNED = "returned"
    ESCALATED = "escalated"
    WITHDRAWN = "withdrawn"
    SUPERSEDED = "superseded"


# 仍可被审核动作推进的版本状态
REVIEWABLE_STATES = {VersionState.PENDING}
# 会话终态：延迟回调到达这些状态时直接忽略
FINAL_SESSION_STATES = {SessionState.ANSWERED, SessionState.ESCALATED, SessionState.CLOSED}


class ReviewError(Exception):
    """业务规则拒绝。"""


class ConflictError(ReviewError):
    """并发或重复操作导致的状态冲突。"""


class PermissionError_(ReviewError):
    """角色权限不足。"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Evidence:
    """引用依据（指南/文献），带有效期与作废状态。"""
    evidence_id: str
    title: str
    valid_until: str           # ISO 时间，过期即不可作为发布依据
    status: str = "active"     # active | superseded（被新证据否定）

    def is_usable(self, now: str) -> bool:
        return self.status == "active" and self.valid_until > now


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    note: str = ""


@dataclass(frozen=True)
class Session:
    session_id: str
    question: str              # 已脱敏的提问
    risk_labels: tuple[str, ...]
    state: str = SessionState.OPEN.value
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class AnswerVersion:
    version_id: str
    session_id: str
    seq: int
    content: str
    citations: tuple[Citation, ...]
    state: str = VersionState.PENDING.value
    created_by: str = ""
    created_at: str = ""
    decided_by: str = ""
    decided_at: str = ""
    decision_reason: str = ""


@dataclass(frozen=True)
class GateDecision:
    """规则判定结果：escalate=直接转人工；doctor=需医生确认；reviewer=审核员可发布。"""
    route: str                 # escalate | doctor | reviewer
    reasons: tuple[str, ...] = ()


def decide_route(risk_labels: tuple[str, ...]) -> GateDecision:
    """由风险标签决定路由。急症永远优先转人工。"""
    labels = set(risk_labels)
    if Risk.EMERGENCY.value in labels:
        return GateDecision("escalate", ("命中急症风险标签，必须转人工",))
    if Risk.HIGH.value in labels:
        return GateDecision("doctor", ("高风险回答，发布前需医生确认",))
    return GateDecision("reviewer", ())


def check_publishable(
    session: Session,
    version: AnswerVersion,
    evidence_map: dict[str, Evidence],
    now: str,
    actor_role: str,
) -> None:
    """发布门槛校验；任何一项不满足即抛出对应异常，重复/延迟提交无法绕过。"""
    if session.state != SessionState.OPEN.value:
        raise ConflictError(f"会话状态为 {session.state}，不能发布新版本")
    if version.state != VersionState.PENDING.value:
        raise ConflictError(f"版本状态为 {version.state}，不在待审核状态")
    route = decide_route(session.risk_labels)
    if route.route == "escalate":
        raise ReviewError("急症会话禁止自动发布，必须转人工")
    if route.route == "doctor" and actor_role != "doctor":
        raise PermissionError_("高风险回答必须由医生确认发布")
    stale = [
        c.evidence_id for c in version.citations
        if c.evidence_id not in evidence_map or not evidence_map[c.evidence_id].is_usable(now)
    ]
    if stale:
        raise ReviewError(f"引用依据已过期或被否定，禁止发布: {', '.join(sorted(stale))}")
