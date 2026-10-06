# 智能问答人工复核

面向医疗问答审核团队的复核服务：保存脱敏后的提问、回答版本、引用依据与
风险标签；规则决定必须医生确认或直接转人工的情形；审核者可退回、要求修改、
发布；已发布内容被新证据否定时撤回并通知仍在处理的会话；重复提交与延迟
回调无法绕过确认门槛。

仅使用 Python 标准库，SQLite 本地持久化。

## 目录

- `src/clinical_review/domain.py` — 时钟、脱敏、证据集、风险规则引擎与数据结构
- `src/clinical_review/store.py` — SQLite 表结构、可重入事务、行级条件更新（CAS）
- `src/clinical_review/service.py` — 复核工作流（提交/回调/认领/退回/修改/确认/发布/撤回/回滚/恢复）
- `src/clinical_review/api.py` — 进程内 JSON 请求适配层
- `tests/test_baseline.py` — 基线行为
- `tests/test_acceptance.py` — 六项验收场景（可控时钟 + 可控证据集）

## 核心规则

1. **急症直接转人工**：提问或模型回答命中急症线索（胸痛、呼吸困难、昏迷、
   自杀等）时案件立即 `escalated_human`，不登记自动回复回调；迟到回调只
   记录 `late_answer_ignored`，不能逆转状态。
2. **高风险医生门槛**：孕妇/新生儿/处方药等命中高风险规则，或引用依据
   失效时，版本必须由医生确认；普通审核者的确认只登记意见、不发布。
3. **引用时效**：证据带 `valid_until` 与 `superseded_by`；回答送达、确认、
   发布、回滚各时点都会复核。过期/被取代 → 打 `OUTDATED_CITATION` 标签、
   升级医生门槛并阻断当前确认；已发布版本被新证据否定 → 撤回并通知会话。
4. **并发安全**：案件认领互斥；版本/案件每次状态推进都是
   `UPDATE ... WHERE status IN (...)` 的 CAS，重复回调有
   `pending→done` 的一次性 CAS，重复提交以脱敏文本指纹去重。
5. **重启恢复**：`recover()` 返回等待回答、等待审核、已转人工、活跃会话与
   未投递通知；证据集同样持久化，重启后否定关系与过期判断继续生效。

## 进程内 API

请求形如 `{"action": ..., "actor": {"id": "rv-1", "role": "reviewer"}}`，
成功返回 `{"ok": true, "data": ...}`（`health`/`register` 保持基线裸格式），
业务错误返回 `{"ok": false, "error": {"code", "message"}}`。

| action | 说明 |
|---|---|
| `open_session` / `close_session` | 打开/关闭用户会话 |
| `submit_question` | 提交提问（脱敏、去重、急症即时升级） |
| `register_callback` / `deliver_answer` | 模型回调登记（幂等）与延迟回答送达 |
| `list_queue` / `case_detail` | 审核队列与案件详情（版本、事件流） |
| `claim` / `release_claim` | 认领/释放案件（互斥） |
| `request_changes` | 退回当前版本 |
| `revise` | 按修改意见产生新版本，重走规则与门槛 |
| `reopen_for_revision` | 对已发布/已撤回内容提出修改 |
| `confirm` | 审核者或医生确认（医生确认高风险稿即发布） |
| `publish` | 满足门槛后发布 |
| `withdraw` | 系统/医生撤回已发布内容 |
| `add_evidence` / `supersede_evidence` | 维护证据集与取代关系 |
| `sweep_stale_citations` | 时间推进后批量清扫在审版本的过期引用 |
| `rollback` | 医生回滚到历史上发布过且引用仍有效的版本 |
| `drain_notifications` / `pending_notifications` | 会话通知投递 |
| `recover` | 重启后的未完成工作恢复视图 |

## 运行

```bash
# unittest
PYTHONPATH=src python3 -m unittest discover -s tests

# pytest（pytest.ini 已配置 pythonpath=src）
python3 -m pytest tests -q

python3 -m compileall src
```

验收测试全部使用 `FixedClock`（可设定/推进时间）与显式构造的
`EvidenceCatalog`（有效期、取代关系可控），覆盖：高风险升级、引用过期、
版本回滚、权限隔离、并发审核（多线程争抢认领/发布/回调/提交）、
重启后未完成会话恢复。
