"""问答复核的领域对象、脱敏与风险规则。

本模块只包含纯领域逻辑，不访问数据库，也不直接取系统时间：
所有时间判断都通过 :class:`Clock` 进行，便于验收时用可控时钟复现
“引用过期”“延迟回调”等场景。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------


class Clock:
    """可替换的时钟。"""

    def now(self) -> datetime:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock(Clock):
    """测试用固定时钟，可人工推进。"""

    def __init__(self, start: datetime | str | None = None) -> None:
        if start is None:
            value = datetime(2026, 1, 1, tzinfo=timezone.utc)
        elif isinstance(start, str):
            value = parse_ts(start)
        else:
            value = start
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        self._now = value

    def now(self) -> datetime:
        return self._now

    def advance(self, **kwargs) -> datetime:
        self._now += timedelta(**kwargs)
        return self._now

    def set(self, value: datetime | str) -> None:
        self._now = parse_ts(value) if isinstance(value, str) else value


def parse_ts(value: str) -> datetime:
    """解析 ISO8601 时间，缺省时按 UTC 处理。"""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

# 案件（一个脱敏提问）生命周期状态
CASE_PENDING = "pending_review"        # 有版本等待医生/审核者确认
CASE_AWAITING = "awaiting_answer"      # 问题已登记，模型回答延迟回调中
CASE_CHANGES = "changes_requested"     # 被退回，等待修改后重新提交
CASE_PUBLISHED = "published"           # 已有发布版本
CASE_ESCALATED = "escalated_human"     # 急症，直接转人工，不走自动回复
CASE_WITHDRAWN = "withdrawn"           # 发布版本被新证据否定后撤回

# 版本状态
V_DRAFT = "draft"                      # 问题先到、回答未到
V_PENDING = "pending_review"
V_CHANGES = "changes_requested"
V_PUBLISHED = "published"
V_SUPERSEDED = "superseded"
V_WITHDRAWN = "withdrawn"
V_ESCALATED = "escalated_human"

# 风险标签
TAG_EMERGENCY = "RED_EMERGENCY"
TAG_HIGH_RISK = "HIGH_RISK"
TAG_MEDIUM = "MEDIUM_RISK"
TAG_LOW = "LOW_RISK"
TAG_OUTDATED_CITATION = "OUTDATED_CITATION"

ROUTE_ESCALATE = "escalate_human"
ROUTE_DOCTOR = "doctor_confirmation"
ROUTE_REVIEWER = "reviewer_confirmation"

# 角色
ROLE_REVIEWER = "reviewer"
ROLE_DOCTOR = "doctor"
ROLE_SYSTEM = "system"


class ReviewError(Exception):
    """业务错误基类，``code`` 供 API 层映射。"""

    code = "review_error"


class NotFoundError(ReviewError):
    code = "not_found"


class PermissionDeniedError(ReviewError):
    code = "permission_denied"


class ConflictError(ReviewError):
    """状态已被并发操作改变（CAS 失败）。"""

    code = "conflict"


class StaleCitationError(ReviewError):
    """引用依据过期或被取代，确认门槛不得通过。"""

    code = "stale_citation"


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------

_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_NAME_RE = re.compile(r"(我叫|我是|患者叫|患者是)([一-鿿]{2,4})")
_NAME_STOPWORDS = {"医生", "患者", "病人", "学生", "老师", "护士", "孕妇",
                   "老人", "小孩", "本人", "一个人"}


def mask_phone(match: re.Match) -> str:
    token = match.group(0)
    return token[0] + "*" * (len(token) - 1)


def mask_idcard(match: re.Match) -> str:
    token = match.group(0)
    return token[0] + "*" * (len(token) - 2) + token[-1]


def sanitize_question(text: str) -> str:
    """去掉提问中的直接标识符，只保存脱敏文本。"""
    if not text:
        return text
    masked = _EMAIL_RE.sub("<邮箱>", text)
    masked = _IDCARD_RE.sub(mask_idcard, masked)
    masked = _PHONE_RE.sub(mask_phone, masked)

    def _mask_name(match: re.Match) -> str:
        prefix, name = match.group(1), match.group(2)
        if name in _NAME_STOPWORDS:
            return match.group(0)
        return prefix + "<姓名>"

    return _NAME_RE.sub(_mask_name, masked)


def dedup_fingerprint(question_safe: str) -> str:
    """对脱敏后提问做规范化指纹，用于拦截重复提交。"""
    normalized = re.sub(r"\s+", "", question_safe)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 证据集
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    valid_from: str
    valid_until: str | None = None
    superseded_by: str | None = None
    title: str = ""


class EvidenceCatalog:
    """可控证据集：审核规则只认这里登记过且在有效期内的依据。"""

    def __init__(self, evidences: list[Evidence] | None = None) -> None:
        self._items: dict[str, Evidence] = {}
        for evidence in evidences or []:
            self._items[evidence.evidence_id] = evidence

    def add(self, evidence: Evidence) -> None:
        self._items[evidence.evidence_id] = evidence

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    # 证据被新证据否定时调用：登记取代关系
    def supersede(self, old_id: str, new_id: str) -> None:
        old = self._items[old_id]
        self._items[old_id] = Evidence(
            old.evidence_id, old.valid_from, old.valid_until,
            superseded_by=new_id, title=old.title,
        )


CITATION_OK = "ok"
CITATION_MISSING = "missing"
CITATION_EXPIRED = "expired"
CITATION_SUPERSEDED = "superseded"


def check_citation(evidence: Evidence | None, now: datetime) -> str:
    if evidence is None:
        return CITATION_MISSING
    if evidence.superseded_by:
        return CITATION_SUPERSEDED
    if evidence.valid_until and parse_ts(evidence.valid_until) < now:
        return CITATION_EXPIRED
    return CITATION_OK


# ---------------------------------------------------------------------------
# 风险规则
# ---------------------------------------------------------------------------

# 命中即必须直接转人工的急症描述
EMERGENCY_KEYWORDS = (
    "胸痛", "胸闷", "呼吸困难", "喘不上气", "昏迷", "意识不清", "抽搐",
    "卒中", "中风", "偏瘫", "口角歪斜", "大出血", "呕血", "黑便",
    "剧烈头痛", "过敏性休克", "自杀", "吞药", "农药",
)

# 非急症但必须医生确认的高风险情形
HIGH_RISK_KEYWORDS = (
    "孕妇", "妊娠", "怀孕", "出血", "婴儿", "新生儿", "处方药",
    "抗菌药", "抗生素", "降压药", "胰岛素", "化疗",
)

MEDIUM_RISK_KEYWORDS = ("发热", "咳嗽", "腹泻", "皮疹", "高血压", "糖尿病")


@dataclass(frozen=True)
class RiskAssessment:
    tags: tuple[str, ...]
    route: str
    require_doctor: bool
    reasons: tuple[str, ...] = ()
    citation_status: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tags": list(self.tags),
            "route": self.route,
            "require_doctor": self.require_doctor,
            "reasons": list(self.reasons),
            "citation_status": dict(self.citation_status),
        }


class RiskRuleEngine:
    """规则引擎：急症转人工，高风险医生确认，引用失效加风险标签。"""

    def __init__(self, catalog: EvidenceCatalog | None = None,
                 emergency=EMERGENCY_KEYWORDS,
                 high_risk=HIGH_RISK_KEYWORDS,
                 medium=MEDIUM_RISK_KEYWORDS) -> None:
        self.catalog = catalog or EvidenceCatalog()
        self.emergency = emergency
        self.high_risk = high_risk
        self.medium = medium

    def evaluate(self, question: str, answer: str | None,
                 citations: list[str] | None, now: datetime) -> RiskAssessment:
        text = f"{question or ''}\n{answer or ''}"
        tags: list[str] = []
        reasons: list[str] = []

        hit_emergency = [word for word in self.emergency if word in text]
        if hit_emergency:
            tags.append(TAG_EMERGENCY)
            reasons.append("急症线索:" + ",".join(hit_emergency))
            # 急症优先级最高：直接转人工，不等待自动回复/确认
            return RiskAssessment(tuple(tags), ROUTE_ESCALATE, False,
                                  tuple(reasons))

        if any(word in text for word in self.high_risk):
            tags.append(TAG_HIGH_RISK)
            reasons.append("命中高风险用药/人群规则")
        elif any(word in text for word in self.medium):
            tags.append(TAG_MEDIUM)
        else:
            tags.append(TAG_LOW)

        statuses = {cid: check_citation(self.catalog.get(cid), now)
                    for cid in (citations or [])}
        bad = {cid: state for cid, state in statuses.items()
               if state != CITATION_OK}
        if bad:
            tags.append(TAG_OUTDATED_CITATION)
            reasons.append("引用失效:" + ",".join(f"{cid}:{state}"
                                                  for cid, state in bad.items()))

        require_doctor = TAG_HIGH_RISK in tags or TAG_OUTDATED_CITATION in tags
        route = ROUTE_DOCTOR if require_doctor else ROUTE_REVIEWER
        return RiskAssessment(tuple(tags), route, require_doctor,
                              tuple(reasons), statuses)


# ---------------------------------------------------------------------------
# 持久化数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    case_id: str
    dedup_key: str
    question_safe: str
    status: str
    session_id: str | None = None
    current_version_id: str | None = None
    require_doctor: bool = False
    claimed_by: str | None = None
    stale: bool = False
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class AnswerVersion:
    version_id: str
    case_id: str
    seq: int
    answer_text: str
    citations: tuple[str, ...]
    risk_tags: tuple[str, ...]
    route: str
    require_doctor: bool
    status: str
    reasons: tuple[str, ...] = ()
    created_by: str = ROLE_SYSTEM
    created_at: str = ""
    decided_at: str = ""


@dataclass(frozen=True)
class Event:
    event_id: str
    case_id: str
    version_id: str | None
    kind: str
    actor: str
    payload: str
    created_at: str


@dataclass(frozen=True)
class CallbackRecord:
    callback_id: str
    case_id: str
    version_id: str
    status: str  # pending / done
    created_at: str
    completed_at: str = ""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    status: str  # active / closed
    created_at: str
    closed_at: str = ""


@dataclass(frozen=True)
class Notification:
    notification_id: str
    session_id: str
    case_id: str
    kind: str
    payload: str
    created_at: str
    version_id: str | None = None
    delivered_at: str = ""


# ---------------------------------------------------------------------------
# 基线兼容
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    """基线版本的简易记录，保留给健康检查/登记接口使用。"""

    record_id: str
    owner_id: str
    state: str = "draft"
    created_at: str = ""

    def with_timestamp(self) -> "Record":
        return Record(self.record_id, self.owner_id, self.state,
                      self.created_at or datetime.now(timezone.utc).isoformat())
