"""六项验收场景：全部使用可控时钟与可控证据集。

覆盖：高风险升级、引用过期、版本回滚、权限隔离、并发审核、重启恢复，
以及重复提交/延迟回调不得绕过确认门槛。
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clinical_review.api import handle
from clinical_review.domain import (
    CASE_AWAITING,
    CASE_CHANGES,
    CASE_ESCALATED,
    CASE_PENDING,
    CASE_PUBLISHED,
    CASE_WITHDRAWN,
    CITATION_SUPERSEDED,
    ConflictError,
    Evidence,
    EvidenceCatalog,
    FixedClock,
    PermissionDeniedError,
    StaleCitationError,
    check_citation,
)
from clinical_review.service import Service
from clinical_review.store import Store

T0 = "2026-01-01T00:00:00+00:00"
EV_GOOD = Evidence("EV-GOOD", "2025-01-01T00:00:00+00:00", None,
                   title="长期有效指南")
EV_GOOD2 = Evidence("EV-GOOD2", "2025-01-01T00:00:00+00:00", None,
                    title="另一份长期指南")
EV_EXP = Evidence("EV-EXP", "2025-01-01T00:00:00+00:00",
                  "2026-06-01T00:00:00+00:00", title="会过期的指南")
EV_NEW = Evidence("EV-NEW", "2026-09-01T00:00:00+00:00", None,
                  title="新版指南")


def make_service(db_path: str = ":memory:", clock: FixedClock | None = None,
                 evidences: tuple[Evidence, ...] = ()):
    clock = clock or FixedClock(T0)
    catalog = EvidenceCatalog(list(evidences))
    return Service(Store(db_path), clock=clock, catalog=catalog), clock


def answer_flow(service: Service, case_id: str, question: str, answer: str,
                citations: list[str] | None = None,
                session_id: str | None = None) -> dict:
    service.submit_question(case_id, question, session_id=session_id)
    service.register_callback(f"cb-{case_id}", case_id)
    return service.deliver_answer(f"cb-{case_id}", answer, citations)


class 高风险升级测试(unittest.TestCase):
    def setUp(self):
        self.service, self.clock = make_service(
            evidences=(EV_GOOD, EV_EXP))

    def test_急症提问立即转人工且不接受回调(self):
        result = self.service.submit_question(
            "c-em", "我突然胸痛、呼吸困难，满头大汗怎么办",
            session_id="s-em")
        self.assertEqual(result["status"], CASE_ESCALATED)
        self.assertIsNone(result["current_version_id"])

        # 自动回复回调不得登记：急症必须等人工
        with self.assertRaises(ConflictError):
            self.service.register_callback("cb-late", "c-em")

        # 会话收到升级通知
        notes = self.service.pending_notifications("s-em")
        self.assertTrue(any(n["kind"] == "escalated_human" for n in notes))

        # 恢复队列中能看到该急症案件
        rec = self.service.recover()
        self.assertEqual([c["case_id"] for c in rec["escalated_human"]],
                         ["c-em"])

    def test_急症在模型回答中暴露也要升级(self):
        self.service.submit_question("c-em2", "今天有点不舒服")
        self.service.register_callback("cb-em2", "c-em2")
        result = self.service.deliver_answer(
            "cb-em2", "你描述的剧烈头痛伴呕吐需要按卒中处理", ["EV-GOOD"])
        self.assertEqual(result["status"], CASE_ESCALATED)

    def test_高风险必须医生确认_审核者不能放行(self):
        result = answer_flow(
            self.service, "c-hi", "孕妇咳嗽能吃什么药",
            "建议多饮水，用药需谨慎", ["EV-GOOD"])
        self.assertEqual(result["status"], CASE_PENDING)
        self.assertTrue(result["version"]["require_doctor"])
        self.assertIn("HIGH_RISK", result["version"]["risk_tags"])

        self.service.claim("c-hi", "rv-1", "reviewer")
        # 审核者尝试直接发布：被医生门槛挡住
        with self.assertRaises(ConflictError):
            self.service.publish("c-hi", "rv-1", "reviewer")
        # 审核者确认只记录意见，不发布，并释放认领
        ack = self.service.confirm("c-hi", "rv-1", "reviewer")
        self.assertFalse(ack["published"])
        self.assertEqual(ack["gate"], "waiting_doctor")
        self.assertEqual(
            self.service.case_detail("c-hi")["status"], CASE_PENDING)

        # 医生认领并确认后才发布
        self.service.claim("c-hi", "dr-1", "doctor")
        published = self.service.confirm("c-hi", "dr-1", "doctor")
        self.assertTrue(published["published"])
        self.assertEqual(published["status"], CASE_PUBLISHED)

    def test_普通风险审核者即可发布(self):
        result = answer_flow(
            self.service, "c-lo", "孩子轻微咳嗽怎么护理",
            "注意休息、多饮水", ["EV-GOOD"])
        self.assertFalse(result["version"]["require_doctor"])
        self.service.claim("c-lo", "rv-2", "reviewer")
        published = self.service.confirm("c-lo", "rv-2", "reviewer")
        self.assertTrue(published["published"])

    def test_重复回调不能重复生效(self):
        answer_flow(self.service, "c-dup", "孩子咳嗽怎么办",
                    "首次回答", ["EV-GOOD"])
        again = self.service.deliver_answer(
            "cb-c-dup", "迟到的不同回答", ["EV-GOOD"])
        self.assertTrue(again["duplicate"])
        detail = self.service.case_detail("c-dup")
        self.assertEqual(len(detail["versions"]), 1)
        self.assertEqual(detail["versions"][0]["answer_text"], "首次回答")

    def test_重复提交被指纹拦截(self):
        first = self.service.submit_question(
            "c-fp", "我是张大宝，电话13800001111，孩子咳嗽怎么办",
            session_id="s-1")
        self.assertFalse(first["duplicate"])
        # 不同直接标识符、脱敏后相同的提问
        second = self.service.submit_question(
            "c-fp2", "我是李小四，电话13911112222，孩子咳嗽怎么办")
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["case_id"], "c-fp")
        self.assertNotIn("13800001111", second["question_safe"])
        self.assertNotIn("张大宝", second["question_safe"])

    def test_急症后的迟到回调不能逆转状态(self):
        self.service.submit_question("c-em3", "突然昏迷、抽搐")
        # 回答送达时才暴露急症的案件：重复送达保持人工升级
        self.service.submit_question("c-em4", "今天有点不舒服")
        self.service.register_callback("cb-em4", "c-em4")
        self.service.deliver_answer("cb-em4", "胸痛可能是心梗", ["EV-GOOD"])
        again = self.service.deliver_answer("cb-em4", "普通建议", ["EV-GOOD"])
        self.assertTrue(again["duplicate"])
        self.assertEqual(
            self.service.case_detail("c-em4")["status"], CASE_ESCALATED)


class 引用过期测试(unittest.TestCase):
    def setUp(self):
        self.service, self.clock = make_service(
            evidences=(EV_GOOD, EV_EXP))

    def test_确认时刻引用过期则阻断并升级门槛(self):
        result = answer_flow(
            self.service, "c-exp", "孩子咳嗽怎么护理",
            "按旧指南护理", ["EV-EXP"])
        # 送达时证据仍有效
        self.assertEqual(result["status"], CASE_PENDING)
        self.assertFalse(result["version"]["require_doctor"])

        self.service.claim("c-exp", "rv-1", "reviewer")
        # 时间推进到证据失效之后
        self.clock.set("2026-07-01T00:00:00+00:00")
        with self.assertRaises(StaleCitationError):
            self.service.confirm("c-exp", "rv-1", "reviewer")

        detail = self.service.case_detail("c-exp")
        self.assertTrue(detail["stale"])
        self.assertTrue(detail["require_doctor"])
        self.assertIsNone(detail["claimed_by"])
        self.assertIn("OUTDATED_CITATION",
                      detail["versions"][0]["risk_tags"])

        # 医生也不能带着失效引用放行
        self.service.claim("c-exp", "dr-1", "doctor")
        with self.assertRaises(StaleCitationError):
            self.service.confirm("c-exp", "dr-1", "doctor")

        # 退回 -> 用有效证据修订 -> 重新审核后发布
        self.service.request_changes("c-exp", "dr-1", "doctor",
                                     "引用已过期，请更新")
        self.service.revise("c-exp", "dr-1", "按新指南护理", ["EV-GOOD"])
        self.service.claim("c-exp", "rv-2", "reviewer")
        published = self.service.confirm("c-exp", "rv-2", "reviewer")
        self.assertTrue(published["published"])

    def test_定时清扫批量拦截过期引用(self):
        answer_flow(self.service, "c-a", "咳嗽怎么办", "回答A", ["EV-EXP"])
        answer_flow(self.service, "c-b", "腹泻怎么办", "回答B", ["EV-GOOD"])
        self.clock.set("2026-07-01T00:00:00+00:00")
        swept = self.service.sweep_stale_citations()
        self.assertEqual(swept["blocked"], ["c-a"])
        self.assertTrue(self.service.case_detail("c-a")["require_doctor"])
        self.assertFalse(self.service.case_detail("c-b")["require_doctor"])

    def test_证据被取代立即判定失效(self):
        result = answer_flow(
            self.service, "c-sup", "咳嗽怎么办", "回答", ["EV-EXP"])
        version_id = result["version"]["version_id"]
        # 尚未过期但被新证据取代
        self.service.add_evidence(EV_NEW)
        self.service.supersede_evidence("EV-EXP", "EV-NEW")
        self.assertEqual(
            check_citation(self.service.catalog.get("EV-EXP"),
                           self.clock.now()),
            CITATION_SUPERSEDED)
        self.service.claim("c-sup", "rv-1", "reviewer")
        with self.assertRaises(StaleCitationError):
            self.service.confirm("c-sup", "rv-1", "reviewer")
        self.assertEqual(
            self.service.store.get_version(version_id).risk_tags[-1],
            "OUTDATED_CITATION")


class 版本回滚测试(unittest.TestCase):
    def setUp(self):
        self.service, self.clock = make_service(
            evidences=(EV_GOOD, EV_GOOD2, EV_NEW))

    def _publish_v1_then_v2(self):
        answer_flow(self.service, "c-rb", "咳嗽怎么办",
                    "建议第一版", ["EV-GOOD"], session_id="s-rb")
        self.service.claim("c-rb", "rv-1", "reviewer")
        self.service.confirm("c-rb", "rv-1", "reviewer")
        v1 = self.service.case_detail("c-rb")["versions"][0]["version_id"]

        # 对已发布内容提出修改，再发布第二版
        self.service.reopen_for_revision("c-rb", "rv-1", "reviewer",
                                         "需要更新措辞")
        self.service.revise("c-rb", "rv-1", "建议第二版", ["EV-GOOD2"])
        self.service.claim("c-rb", "rv-1", "reviewer")
        self.service.confirm("c-rb", "rv-1", "reviewer")
        detail = self.service.case_detail("c-rb")
        v2 = detail["current_version_id"]
        self.assertEqual(detail["versions"][0]["status"], "superseded")
        self.assertEqual(detail["versions"][1]["status"], "published")
        return v1, v2

    def test_新证据否定发布版本_撤回并通知_医生可回滚(self):
        v1, v2 = self._publish_v1_then_v2()

        outcome = self.service.supersede_evidence("EV-GOOD2", "EV-NEW",
                                                  reason="新证据否定")
        self.assertEqual(outcome["published_withdrawn"], ["c-rb"])
        detail = self.service.case_detail("c-rb")
        self.assertEqual(detail["status"], CASE_WITHDRAWN)
        self.assertEqual(self.service.store.get_version(v2).status,
                         "withdrawn")

        # 仍在处理的会话收到撤回通知
        kinds = [n["kind"] for n in
                 self.service.pending_notifications("s-rb")]
        self.assertIn("answer_withdrawn", kinds)

        # 非医生不能回滚；引用失效的版本不能作为回滚目标
        with self.assertRaises(PermissionDeniedError):
            self.service.rollback_to("c-rb", "rv-1", "reviewer", v1)
        rolled = self.service.rollback_to("c-rb", "dr-1", "doctor", v1)
        self.assertEqual(rolled["status"], CASE_PUBLISHED)
        self.assertEqual(rolled["current_version_id"], v1)
        self.assertEqual(
            self.service.store.get_version(v1).status, "published")
        kinds = [n["kind"] for n in
                 self.service.pending_notifications("s-rb")]
        self.assertIn("answer_rollback", kinds)

        # 回滚版本日后再被否定，仍会撤回
        self.service.supersede_evidence("EV-GOOD", "EV-NEW")
        self.assertEqual(
            self.service.case_detail("c-rb")["status"], CASE_WITHDRAWN)

    def test_只能回滚到历史发布版本且引用须有效(self):
        v1, v2 = self._publish_v1_then_v2()
        self.service.supersede_evidence("EV-GOOD2", "EV-NEW")
        # v2 当前是 withdrawn 且引用失效：不能回滚到它
        with self.assertRaises(StaleCitationError):
            self.service.rollback_to("c-rb", "dr-1", "doctor", v2)


class 权限隔离测试(unittest.TestCase):
    def setUp(self):
        self.service, _ = make_service(evidences=(EV_GOOD,))

    def test_角色边界(self):
        answer_flow(self.service, "c-p", "孩子咳嗽怎么办", "回答", ["EV-GOOD"])
        self.service.claim("c-p", "rv-1", "reviewer")
        with self.assertRaises(PermissionDeniedError):
            self.service.confirm("c-p", "sys", "system")
        with self.assertRaises(PermissionDeniedError):
            self.service.request_changes("c-p", "sys", "system")
        with self.assertRaises(PermissionDeniedError):
            self.service.withdraw_published("c-p", "x",
                                            actor="rv-9",
                                            actor_role="reviewer")

    def test_认领互斥_未认领者不能操作(self):
        answer_flow(self.service, "c-lock", "孩子咳嗽怎么办", "回答",
                    ["EV-GOOD"])
        self.service.claim("c-lock", "rv-1", "reviewer")
        with self.assertRaises(ConflictError):
            self.service.claim("c-lock", "rv-2", "reviewer")
        with self.assertRaises(PermissionDeniedError):
            self.service.confirm("c-lock", "rv-2", "reviewer")
        with self.assertRaises(PermissionDeniedError):
            self.service.request_changes("c-lock", "rv-2", "reviewer")
        # 释放后其他人可以接手
        self.service.release_claim("c-lock", "rv-1")
        self.service.claim("c-lock", "rv-2", "reviewer")
        self.assertTrue(
            self.service.confirm("c-lock", "rv-2", "reviewer")["published"])

    def test_脱敏落库_邮箱证件手机姓名(self):
        raw = ("我是张大宝，邮箱 dabao@example.com，手机13800001234，"
               "身份证11010119900307123X，孩子咳嗽")
        result = self.service.submit_question("c-mask", raw)
        safe = result["question_safe"]
        self.assertNotIn("dabao@example.com", safe)
        self.assertNotIn("13800001234", safe)
        self.assertNotIn("11010119900307123X", safe)
        self.assertNotIn("张大宝", safe)
        self.assertIn("孩子咳嗽", safe)
        self.assertIn("<邮箱>", safe)
        self.assertIn("<姓名>", safe)

    def test_API错误信封与角色(self):
        answer_flow(self.service, "c-api", "孩子咳嗽怎么办", "回答",
                    ["EV-GOOD"])
        denied = json.loads(handle(json.dumps({
            "action": "rollback", "case_id": "c-api",
            "target_version_id": "c-api:v1",
            "actor": {"id": "rv-1", "role": "reviewer"},
        }), self.service))
        self.assertFalse(denied["ok"])
        self.assertEqual(denied["error"]["code"], "permission_denied")
        ok = json.loads(handle(json.dumps({"action": "health"}),
                               self.service))
        self.assertEqual(ok["status"], "ok")


class 并发审核测试(unittest.TestCase):
    def setUp(self):
        self.clock = FixedClock(T0)
        self.catalog = EvidenceCatalog([EV_GOOD])
        self.store = Store(":memory:")
        self.services = [
            Service(self.store, clock=self.clock, catalog=self.catalog)
            for _ in range(6)
        ]
        for i in range(12):
            answer_flow(self.services[0], f"c-{i}",
                        f"并发咳嗽问题{i}", f"回答{i}", ["EV-GOOD"])

    def tearDown(self):
        self.store.close()

    def test_多审核者并发只发布一次(self):
        errors: list[Exception] = []

        def worker(svc: Service, idx: int) -> None:
            actor = f"rv-{idx}"
            for _ in range(200):
                pending = svc.list_queue((CASE_PENDING,))
                if not pending:
                    return
                claimed = None
                for case in pending:
                    try:
                        svc.claim(case["case_id"], actor, "reviewer")
                        claimed = case["case_id"]
                        break
                    except (ConflictError, PermissionDeniedError):
                        continue
                if claimed is None:
                    continue
                try:
                    svc.confirm(claimed, actor, "reviewer")
                except (ConflictError, PermissionDeniedError,
                        StaleCitationError) as exc:
                    errors.append(exc)

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(worker, self.services, range(6)))

        for i in range(12):
            detail = self.services[0].case_detail(f"c-{i}")
            self.assertEqual(detail["status"], CASE_PUBLISHED)
            published_events = [e for e in detail["events"]
                                if e["kind"] == "published"]
            self.assertEqual(len(published_events), 1,
                             f"c-{i} 被重复发布")

    def test_同一回调并发送达只生效一次(self):
        svc = self.services[0]
        svc.submit_question("c-cb", "并发腹泻问题")
        svc.register_callback("cb-cb", "c-cb")
        barrier = threading.Barrier(6)
        results: list[dict] = []

        def deliver(_idx: int) -> None:
            barrier.wait()
            results.append(svc.deliver_answer(
                "cb-cb", "唯一回答", ["EV-GOOD"]))

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(deliver, range(6)))

        winners = [r for r in results if not r.get("duplicate")]
        self.assertEqual(len(winners), 1)
        detail = svc.case_detail("c-cb")
        self.assertEqual(len(detail["versions"]), 1)
        self.assertEqual(detail["versions"][0]["answer_text"], "唯一回答")

    def test_同一提问并发提交只建一个案件(self):
        svc = self.services[0]
        barrier = threading.Barrier(6)
        outcomes: list[dict] = []

        def submit(idx: int) -> None:
            barrier.wait()
            outcomes.append(svc.submit_question(
                f"c-race-{idx}", "并发的同一皮疹问题怎么办"))

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(submit, range(6)))

        firsts = [o for o in outcomes if not o["duplicate"]]
        self.assertEqual(len(firsts), 1)
        self.assertEqual(len(svc.list_queue()),
                         12 + 1)  # setUp 的 12 个 + 1 个新案件

    def test_高风险案件审核者与医生争抢_仍须医生放行(self):
        svc = self.services[0]
        answer_flow(svc, "c-hip", "孕妇发烧怎么办", "回答", ["EV-GOOD"])

        def reviewer() -> None:
            try:
                svc.claim("c-hip", "rv-x", "reviewer")
                svc.confirm("c-hip", "rv-x", "reviewer")
            except (ConflictError, PermissionDeniedError):
                pass

        t = threading.Thread(target=reviewer)
        t.start()
        t.join()
        # 无论审核者是否抢先，医生完成确认前不得发布
        self.assertEqual(svc.case_detail("c-hip")["status"], CASE_PENDING)
        svc.claim("c-hip", "dr-x", "doctor")
        self.assertTrue(
            svc.confirm("c-hip", "dr-x", "doctor")["published"])


class 重启恢复测试(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "review.db")
        self.clock = FixedClock(T0)
        self.service = Service(
            Store(self.db_path), clock=self.clock,
            catalog=EvidenceCatalog([EV_GOOD, EV_GOOD2, EV_EXP]))
        # a) 等待模型回答（回调已登记未送达）
        self.service.open_session("s-1")
        self.service.submit_question("c-wait", "孩子鼻塞怎么办",
                                     session_id="s-1")
        self.service.register_callback("cb-wait", "c-wait")
        # b) 已送达、等待人工审核
        answer_flow(self.service, "c-pend", "孩子咳嗽怎么办",
                    "待审回答", ["EV-GOOD"], session_id="s-1")
        self.service.claim("c-pend", "rv-1", "reviewer")
        # c) 急症转人工
        self.service.submit_question("c-em", "胸痛呼吸困难",
                                     session_id="s-1")
        # d) 已发布
        answer_flow(self.service, "c-pub", "轻微皮疹怎么办",
                    "已发布回答", ["EV-GOOD2"], session_id="s-2")
        self.service.claim("c-pub", "rv-1", "reviewer")
        self.service.confirm("c-pub", "rv-1", "reviewer")
        # 已关闭会话：后续通知不应再投递给它
        self.service.open_session("s-closed")
        self.service.close_session("s-closed")

    def tearDown(self):
        self.service.store.close()
        self.tmp.cleanup()

    def _restart(self, at: str = "2026-01-02T00:00:00+00:00"):
        self.service.store.close()
        clock = FixedClock(at)
        service = Service(Store(self.db_path), clock=clock)
        return service, clock

    def test_重启后恢复未完成工作与未投递通知(self):
        service, clock = self._restart()
        state = service.recover()

        awaiting = {item["case_id"] for item in state["awaiting_answer"]}
        self.assertEqual(awaiting, {"c-wait"})
        pending = {c["case_id"] for c in state["pending_review"]}
        self.assertEqual(pending, {"c-pend"})
        escalated = {c["case_id"] for c in state["escalated_human"]}
        self.assertEqual(escalated, {"c-em"})
        sessions = {s["session_id"] for s in state["active_sessions"]}
        self.assertEqual(sessions, {"s-1", "s-2"})
        self.assertNotIn("s-closed", sessions)
        # 升级通知与发布通知在重启后仍可投递
        undelivered = {(n["session_id"], n["kind"])
                       for n in state["undelivered_notifications"]}
        self.assertIn(("s-1", "escalated_human"), undelivered)
        self.assertIn(("s-2", "answer_published"), undelivered)
        # 证据集（含有效期）一并恢复
        self.assertEqual(
            service.catalog.get("EV-EXP").valid_until, EV_EXP.valid_until)

        # 重启后迟到的回调可以继续被处理
        delivered = service.deliver_answer(
            "cb-wait", "重启后到达的回答", ["EV-GOOD"])
        self.assertEqual(delivered["status"], CASE_PENDING)
        # 重复送达依旧无效
        self.assertTrue(service.deliver_answer(
            "cb-wait", "重复回答", ["EV-GOOD"])["duplicate"])

        # 通知投递一次后消失
        s1_notes = service.drain_notifications("s-1")
        self.assertTrue(any(n["kind"] == "escalated_human"
                            for n in s1_notes))
        self.assertEqual(service.drain_notifications("s-1"), [])

    def test_重启后新证据否定仍然撤回并只通知活跃会话(self):
        service, clock = self._restart()
        service.add_evidence(EV_NEW)
        outcome = service.supersede_evidence("EV-GOOD2", "EV-NEW")
        self.assertEqual(outcome["published_withdrawn"], ["c-pub"])
        self.assertEqual(
            service.case_detail("c-pub")["status"], CASE_WITHDRAWN)
        notified_sessions = {n.session_id for n in
                             service.store.list_undelivered()}
        self.assertIn("s-2", notified_sessions)
        self.assertNotIn("s-closed", notified_sessions)

    def test_重启后引用过期门槛仍然生效(self):
        # EV_EXP 有效期到 2026-06-01；重启到 7 月
        service, clock = self._restart("2026-07-01T00:00:00+00:00")
        answer_flow(service, "c-old", "腹泻怎么办", "旧依据回答", ["EV-EXP"])
        service.claim("c-old", "rv-9", "reviewer")
        with self.assertRaises(StaleCitationError):
            service.confirm("c-old", "rv-9", "reviewer")
        self.assertTrue(service.case_detail("c-old")["stale"])


if __name__ == "__main__":
    unittest.main()
