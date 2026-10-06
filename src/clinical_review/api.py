"""供进程内调用的轻量请求适配层。

所有请求形如 ``{"action": ..., ...}``，返回 JSON 字符串。
业务错误统一返回 ``{"ok": false, "error": {"code", "message"}}``，
不向调用方泄漏异常堆栈。
"""
from __future__ import annotations

import json

from .domain import (
    ConflictError,
    Evidence,
    NotFoundError,
    PermissionDeniedError,
    ReviewError,
    StaleCitationError,
)
from .service import Service
from .store import Store


def build_service(db_path: str = ":memory:", clock=None,
                  evidences: list[Evidence] | None = None) -> Service:
    """应用启动入口：同一文件 DB 重启后数据仍在，可直接 ``recover()``。"""
    service = Service(Store(db_path), clock=clock)
    for evidence in evidences or []:
        service.add_evidence(evidence)
    return service


def _actor(body: dict) -> tuple[str, str]:
    actor = body.get("actor") or {}
    if isinstance(actor, str):
        return actor, "reviewer"
    return str(actor.get("id", "anonymous")), str(actor.get("role", "reviewer"))


def handle(payload: str, service: Service | None = None) -> str:
    service = service or Service()
    try:
        body = json.loads(payload)
        action = body.get("action")
        result = _dispatch(action, body, service)
    except ReviewError as exc:
        return json.dumps({"ok": False,
                           "error": {"code": exc.code, "message": str(exc)}},
                          ensure_ascii=False)
    except KeyError as exc:
        return json.dumps({"ok": False,
                           "error": {"code": "bad_request",
                                     "message": f"缺少字段: {exc.args[0]}"}},
                          ensure_ascii=False)
    except (ValueError, json.JSONDecodeError) as exc:
        return json.dumps({"ok": False,
                           "error": {"code": "bad_request",
                                     "message": str(exc)}},
                          ensure_ascii=False)
    # health/register 保持基线裸响应格式
    if action in ("health", "register"):
        return json.dumps(result, ensure_ascii=False)
    return json.dumps({"ok": True, "data": result}, ensure_ascii=False)


def _dispatch(action: str, body: dict, service: Service) -> dict:
    if action == "health":
        return service.health()
    if action == "register":
        return service.register(str(body["record_id"]), str(body["owner_id"]))

    if action == "open_session":
        return service.open_session(str(body["session_id"]))
    if action == "close_session":
        return service.close_session(str(body["session_id"]))

    if action == "submit_question":
        return service.submit_question(
            str(body["case_id"]), str(body["question"]),
            session_id=body.get("session_id"))

    if action == "register_callback":
        return service.register_callback(str(body["callback_id"]),
                                         str(body["case_id"]))
    if action == "deliver_answer":
        return service.deliver_answer(
            str(body["callback_id"]), str(body["answer_text"]),
            citations=list(body.get("citations") or []))

    if action == "list_queue":
        statuses = body.get("statuses")
        return {"cases": service.list_queue(tuple(statuses) if statuses else None)}

    if action == "case_detail":
        return service.case_detail(str(body["case_id"]))

    if action == "claim":
        actor_id, role = _actor(body)
        return service.claim(str(body["case_id"]), actor_id, role)
    if action == "release_claim":
        actor_id, _ = _actor(body)
        return service.release_claim(str(body["case_id"]), actor_id)
    if action == "request_changes":
        actor_id, role = _actor(body)
        return service.request_changes(str(body["case_id"]), actor_id, role,
                                       str(body.get("comment", "")))
    if action == "reopen_for_revision":
        actor_id, role = _actor(body)
        return service.reopen_for_revision(str(body["case_id"]), actor_id,
                                           role, str(body.get("comment", "")))
    if action == "revise":
        actor_id, _ = _actor(body)
        return service.revise(str(body["case_id"]), actor_id,
                              str(body["answer_text"]),
                              citations=list(body.get("citations") or []),
                              comment=str(body.get("comment", "")))
    if action == "confirm":
        actor_id, role = _actor(body)
        return service.confirm(str(body["case_id"]), actor_id, role,
                               str(body.get("comment", "")))
    if action == "publish":
        actor_id, role = _actor(body)
        return service.publish(str(body["case_id"]), actor_id, role)
    if action == "withdraw":
        return service.withdraw_published(
            str(body["case_id"]), str(body.get("reason", "")),
            actor=str(body.get("actor_id", "system")),
            actor_role=str(body.get("actor_role", "system")))
    if action == "rollback":
        actor_id, role = _actor(body)
        return service.rollback_to(str(body["case_id"]), actor_id, role,
                                   str(body["target_version_id"]))

    if action == "add_evidence":
        data = body["evidence"]
        service.add_evidence(Evidence(
            evidence_id=str(data["evidence_id"]),
            valid_from=str(data["valid_from"]),
            valid_until=data.get("valid_until"),
            superseded_by=data.get("superseded_by"),
            title=str(data.get("title", ""))))
        return {"added": data["evidence_id"]}
    if action == "supersede_evidence":
        return service.supersede_evidence(str(body["old_id"]),
                                          str(body["new_id"]),
                                          str(body.get("reason", "新证据否定")))
    if action == "sweep_stale_citations":
        return service.sweep_stale_citations()

    if action == "drain_notifications":
        return {"notifications":
                service.drain_notifications(str(body["session_id"]))}
    if action == "pending_notifications":
        return {"notifications":
                service.pending_notifications(str(body["session_id"]))}
    if action == "recover":
        return service.recover()

    raise ValueError("不支持的请求动作")
