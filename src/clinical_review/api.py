"""供进程内调用的轻量请求适配层：JSON 请求 -> 服务调用 -> JSON 响应。

业务异常以 {"error": ..., "type": ...} 返回，不抛出给调用方。
"""
import json

from .domain import ConflictError, PermissionError_, ReviewError
from .service import Service


def handle(payload: str, service: Service | None = None) -> str:
    service = service or Service()
    body = json.loads(payload)
    action = body.get("action")
    try:
        if action == "health":
            result = service.health()
        elif action == "register_evidence":
            result = service.register_evidence(
                str(body["evidence_id"]), str(body["title"]), str(body["valid_until"]))
        elif action == "supersede_evidence":
            result = service.supersede_evidence(
                str(body["evidence_id"]), str(body["actor"]), str(body["role"]),
                str(body.get("reason", "")))
        elif action == "create_session":
            result = service.create_session(
                str(body["session_id"]), str(body["question"]),
                list(body.get("risk_labels", [])), body.get("idempotency_key"))
        elif action == "submit_answer":
            result = service.submit_answer(
                str(body["session_id"]), str(body["version_id"]), str(body["content"]),
                list(body.get("citations", [])), str(body["actor"]),
                str(body.get("role", "reviewer")), body.get("idempotency_key"))
        elif action == "propose_edit":
            result = service.propose_edit(
                str(body["version_id"]), str(body["new_version_id"]), str(body["content"]),
                list(body.get("citations", [])), str(body["actor"]), str(body["role"]))
        elif action == "return":
            result = service.return_version(
                str(body["version_id"]), str(body["actor"]), str(body["role"]),
                str(body.get("reason", "")))
        elif action == "escalate":
            result = service.escalate(
                str(body["version_id"]), str(body["actor"]), str(body["role"]),
                str(body.get("reason", "")))
        elif action == "publish":
            result = service.publish(
                str(body["version_id"]), str(body["actor"]), str(body["role"]))
        elif action == "deliver_callback":
            result = service.deliver_callback(
                str(body["session_id"]), str(body["callback_id"]),
                str(body.get("callback_action", "model_update")))
        elif action == "get_session":
            result = service.get_session(str(body["session_id"]))
        elif action == "list_versions":
            result = service.list_versions(str(body["session_id"]))
        elif action == "list_unfinished":
            result = service.list_unfinished()
        elif action == "list_notifications":
            result = service.list_notifications(str(body["session_id"]))
        elif action == "list_events":
            result = service.list_events(str(body["session_id"]))
        else:
            raise ValueError("不支持的请求动作")
    except (ReviewError, ConflictError, PermissionError_) as exc:
        return json.dumps({"error": str(exc), "type": type(exc).__name__},
                          ensure_ascii=False)
    return json.dumps(result, ensure_ascii=False)
