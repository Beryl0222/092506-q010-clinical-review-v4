"""验收测试：可控时间与证据集下的六大场景。

- 高风险升级：急症会话直接转人工，禁止自动发布
- 引用过期：过期/被否定的证据不能作为发布依据
- 版本回滚：已发布版本被撤回并通知在途会话
- 权限隔离：角色门槛不可绕过
- 并发审核：同一版本只能被一人发布
- 重启恢复：未完成会话在重启后可继续处理
另覆盖幂等提交与延迟回调。
"""
import json
import threading
import unittest

from clinical_review.api import handle
from clinical_review.domain import ConflictError, PermissionError_, ReviewError
from clinical_review.service import Service
from clinical_review.store import Store


class 可控时钟:
    def __init__(self, now="2026-01-01T00:00:00+00:00"):
        self.now = now

    def __call__(self):
        return self.now

    def advance_to(self, now):
        self.now = now


def 建服务(clock=None, store=None):
    return Service(store or Store(), clock or 可控时钟())


def 提交正常会话(service, session_id="s-1", version_id="v-1", evidence="ev-1"):
    service.register_evidence(evidence, "分诊指南", "2027-01-01T00:00:00+00:00")
    service.create_session(session_id, "脱敏后的症状描述", ["normal"])
    return service.submit_answer(session_id, version_id, "建议内容",
                                 [{"evidence_id": evidence}], "model")


class 高风险升级测试(unittest.TestCase):
    def test_急症会话提交后直接转人工且禁止发布(self):
        svc = 建服务()
        svc.create_session("s-1", "胸痛伴呼吸困难", ["emergency"])
        result = svc.submit_answer("s-1", "v-1", "建议自行观察", [], "model")
        self.assertEqual(result["route"], "escalate")
        self.assertEqual(result["state"], "escalated")
        self.assertEqual(svc.get_session("s-1")["state"], "escalated")
        with self.assertRaises(ReviewError):
            svc.publish("v-1", "dr-1", "doctor")
        kinds = [n["kind"] for n in svc.list_notifications("s-1")]
        self.assertIn("escalated", kinds)

    def test_审核中可人工升级(self):
        svc = 建服务()
        提交正常会话(svc)
        svc.escalate("v-1", "rev-1", "reviewer", "症状描述疑似急症")
        self.assertEqual(svc.get_session("s-1")["state"], "escalated")


class 引用过期测试(unittest.TestCase):
    def test_证据过期后禁止发布(self):
        clock = 可控时钟()
        svc = 建服务(clock)
        提交正常会话(svc, evidence="ev-1")
        clock.advance_to("2027-02-01T00:00:00+00:00")  # 超过 valid_until
        with self.assertRaises(ReviewError):
            svc.publish("v-1", "rev-1", "reviewer")

    def test_证据被否定后禁止发布(self):
        svc = 建服务()
        提交正常会话(svc)
        svc.supersede_evidence("ev-1", "dr-1", "doctor", "新指南推翻旧结论")
        with self.assertRaises(ReviewError):
            svc.publish("v-1", "rev-1", "reviewer")

    def test_引用不存在的证据禁止发布(self):
        svc = 建服务()
        svc.create_session("s-1", "问题", ["normal"])
        svc.submit_answer("s-1", "v-1", "回答", [{"evidence_id": "ev-x"}], "model")
        with self.assertRaises(ReviewError):
            svc.publish("v-1", "rev-1", "reviewer")


class 版本回滚测试(unittest.TestCase):
    def test_已发布版本被撤回并通知在途会话(self):
        svc = 建服务()
        提交正常会话(svc, "s-1", "v-1")
        svc.publish("v-1", "rev-1", "reviewer")
        # 另一个仍在处理中的会话引用了同一证据
        svc.create_session("s-2", "另一个问题", ["normal"])
        svc.submit_answer("s-2", "v-2", "回答", [{"evidence_id": "ev-1"}], "model")

        result = svc.supersede_evidence("ev-1", "dr-1", "doctor", "新证据否定")
        self.assertEqual(result["withdrawn_versions"], ["v-1"])
        self.assertEqual(svc.get_session("s-1")["state"], "open")  # 回到待处理
        self.assertEqual(svc.list_versions("s-1")[0]["state"], "withdrawn")
        kinds1 = [n["kind"] for n in svc.list_notifications("s-1")]
        kinds2 = [n["kind"] for n in svc.list_notifications("s-2")]
        self.assertIn("version_withdrawn", kinds1)
        self.assertIn("evidence_superseded", kinds2)

    def test_退回与修改产生新版本(self):
        svc = 建服务()
        提交正常会话(svc)
        svc.return_version("v-1", "rev-1", "reviewer", "表述不清")
        svc.submit_answer("s-1", "v-2", "修改后回答", [{"evidence_id": "ev-1"}], "model")
        svc.propose_edit("v-2", "v-3", "审核者修订稿", [{"evidence_id": "ev-1"}], "rev-1", "reviewer")
        svc.publish("v-3", "rev-1", "reviewer")
        states = [v["state"] for v in svc.list_versions("s-1")]
        self.assertEqual(states, ["returned", "superseded", "published"])
        self.assertEqual(svc.get_session("s-1")["state"], "answered")


class 权限隔离测试(unittest.TestCase):
    def test_高风险必须由医生确认(self):
        svc = 建服务()
        svc.register_evidence("ev-1", "指南", "2027-01-01T00:00:00+00:00")
        svc.create_session("s-1", "问题", ["high"])
        svc.submit_answer("s-1", "v-1", "回答", [{"evidence_id": "ev-1"}], "model")
        with self.assertRaises(PermissionError_):
            svc.publish("v-1", "rev-1", "reviewer")
        svc.publish("v-1", "dr-1", "doctor")
        self.assertEqual(svc.get_session("s-1")["state"], "answered")

    def test_否定证据必须由医生执行(self):
        svc = 建服务()
        svc.register_evidence("ev-1", "指南", "2027-01-01T00:00:00+00:00")
        with self.assertRaises(PermissionError_):
            svc.supersede_evidence("ev-1", "rev-1", "reviewer")

    def test_未知角色被拒绝(self):
        svc = 建服务()
        提交正常会话(svc)
        with self.assertRaises(PermissionError_):
            svc.publish("v-1", "guest-1", "guest")


class 并发审核测试(unittest.TestCase):
    def test_同一版本只能被一人发布(self):
        svc = 建服务()
        提交正常会话(svc)
        results, errors = [], []

        def 发布(actor):
            try:
                results.append(svc.publish("v-1", actor, "reviewer"))
            except ConflictError:
                errors.append(actor)

        threads = [threading.Thread(target=发布, args=(f"rev-{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 7)
        self.assertEqual(svc.list_versions("s-1")[0]["state"], "published")


class 重启恢复测试(unittest.TestCase):
    def test_重启后未完成会话可恢复并继续(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "review.db")
            clock = 可控时钟()
            svc = Service(Store(path), clock)
            提交正常会话(svc, "s-1", "v-1")
            svc.create_session("s-2", "胸痛", ["emergency"])
            svc.submit_answer("s-2", "v-2", "回答", [], "model")
            svc.store.close()

            svc2 = Service(Store(path), clock)  # 模拟重启
            unfinished = {s["session_id"]: s for s in svc2.list_unfinished()}
            self.assertEqual(unfinished["s-1"]["pending_versions"], ["v-1"])
            self.assertEqual(unfinished["s-2"]["state"], "escalated")
            svc2.publish("v-1", "rev-1", "reviewer")  # 恢复后继续处理
            self.assertEqual(svc2.get_session("s-1")["state"], "answered")
            svc2.store.close()


class 幂等与延迟回调测试(unittest.TestCase):
    def test_重复提交不产生重复记录(self):
        svc = 建服务()
        svc.register_evidence("ev-1", "指南", "2027-01-01T00:00:00+00:00")
        svc.create_session("s-1", "问题", ["normal"], idempotency_key="k-s1")
        again = svc.create_session("s-1", "问题", ["normal"], idempotency_key="k-s1")
        self.assertTrue(again["idempotent_replay"])
        svc.submit_answer("s-1", "v-1", "回答", [{"evidence_id": "ev-1"}], "model",
                          idempotency_key="k-v1")
        svc.submit_answer("s-1", "v-1", "回答", [{"evidence_id": "ev-1"}], "model",
                          idempotency_key="k-v1")
        self.assertEqual(len(svc.list_versions("s-1")), 1)

    def test_延迟回调不能绕过确认门槛(self):
        svc = 建服务()
        提交正常会话(svc)
        svc.publish("v-1", "rev-1", "reviewer")
        late = svc.deliver_callback("s-1", "cb-1")  # 会话已 answered 的迟到回调
        self.assertFalse(late["applied"])
        dup = svc.deliver_callback("s-1", "cb-1")   # 重复回调
        self.assertFalse(dup["applied"])
        self.assertEqual(svc.list_versions("s-1")[0]["state"], "published")

    def test_在途会话回调只生效一次(self):
        svc = 建服务()
        提交正常会话(svc)
        self.assertTrue(svc.deliver_callback("s-1", "cb-1")["applied"])
        self.assertFalse(svc.deliver_callback("s-1", "cb-1")["applied"])


class API适配测试(unittest.TestCase):
    def test_完整流程经API(self):
        svc = 建服务()

        def call(body):
            return json.loads(handle(json.dumps(body), svc))

        self.assertEqual(call({"action": "health"})["status"], "ok")
        call({"action": "register_evidence", "evidence_id": "ev-1",
              "title": "指南", "valid_until": "2027-01-01T00:00:00+00:00"})
        call({"action": "create_session", "session_id": "s-1",
              "question": "问题", "risk_labels": ["high"]})
        call({"action": "submit_answer", "session_id": "s-1", "version_id": "v-1",
              "content": "回答", "citations": [{"evidence_id": "ev-1"}], "actor": "model"})
        denied = call({"action": "publish", "version_id": "v-1",
                       "actor": "rev-1", "role": "reviewer"})
        self.assertEqual(denied["type"], "PermissionError_")
        ok = call({"action": "publish", "version_id": "v-1",
                   "actor": "dr-1", "role": "doctor"})
        self.assertEqual(ok["state"], "published")


if __name__ == "__main__":
    unittest.main()
