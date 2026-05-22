# ODCR_AUDIT_DELIVERY_SKILL

## 1. 适用范围

本 skill 适用于 ODCR 的：

- 审计；
- 修复；
- 重构；
- 实验；
- 运行；
- 指标聚合；
- 提示词生成；
- 上下文接管；
- 执行结果分析。

后续提示词只需要写：

“输出与归档遵循 docs/skills/ODCR_AUDIT_DELIVERY_SKILL.md。”

不要在每个任务里复制完整报告模板。

## 2. 会话约束

执行期间中途不允许多次状态更新。
最终结果必须一次性完整交付。
严禁连续播报执行进度，避免触发长会话自动压缩。
本阶段中间结果、搜索命中、证据、日志、阶段摘要、最终报告必须写入 AI_analysis。
不要把“凑合能跑”误判为“符合新架构规范”。
不要为了代码完美建议无必要大重构。
只有证据表明不重构无法收口时，才建议重构。

## 3. 固定目录

仓库根目录：

`/public/home/zhangliml/lc/ODCR/ODCR-main`

辅助目录：

`/public/home/zhangliml/lc/ODCR/ODCR-main/AI_analysis`

必须确保：

- `AI_analysis/`
- `AI_analysis/00_index/`
- `AI_analysis/01_raw_logs/`
- `AI_analysis/02_search_hits/`
- `AI_analysis/03_evidence_ledgers/`
- `AI_analysis/04_phase_summaries/`
- `AI_analysis/05_final_reports/`

## 4. 每轮任务必须定义 TASK_NAME

每轮任务必须有固定 TASK_NAME，例如：

`step5_explanation_only_gpu_preflight`

所有输出文件按 TASK_NAME 命名。

## 5. 必须写入的辅助文件

原始执行日志：

`AI_analysis/01_raw_logs/audit_${TASK_NAME}.log`

关键搜索/命中结果：

`AI_analysis/02_search_hits/audit_${TASK_NAME}_hits.txt`

证据账本：

`AI_analysis/03_evidence_ledgers/audit_${TASK_NAME}_ledger.md`

阶段摘要：

`AI_analysis/04_phase_summaries/audit_${TASK_NAME}_summary.md`

最终报告：

`AI_analysis/05_final_reports/audit_${TASK_NAME}_report.md`

机器裁决 JSON：

`AI_analysis/05_final_reports/audit_${TASK_NAME}_machine_verdict.json`

可选索引：

`AI_analysis/00_index/audit_${TASK_NAME}_index.md`

## 6. 标准任务结构

每个 ODCR 任务提示词应包含：

- 阶段名；
- 任务目标；
- 当前已知事实；
- 本轮范围；
- 必须回答的问题；
- 允许修改范围；
- 禁止事项；
- 执行顺序；
- 验证命令；
- 最终报告格式；
- machine verdict JSON 字段；
- 裁决标准。

## 7. 审计任务禁止事项

审计任务中：

- 不要顺手开始改代码；
- 不要顺手写下一阶段提示词；
- 不要重新审查与本次审计无关的业务逻辑；
- 不要把历史 AI_analysis 当 active truth；
- 不要把旧 run 当新架构结果；
- 不要把 dry-run / smoke / diagnostic 当 formal evidence；
- 不要用 quality_audit.json 覆盖 stage_status/eval_handoff truth。

## 8. 修复任务禁止事项

修复任务中：

- 不要保留旧 active 路径；
- 不要用 alias 掩盖已退役逻辑；
- 不要把旧评分、旧推荐、旧 scorer-only/rating-only 路径改成“禁用但还在代码里”的包装；确认退役后要物理删除，
  只允许 Step3 rating_source/eval_handoff 作为指标引用继续存在。
- 不要把 GPU/bridge blocker 当成结论。用户明确授权的 fixed-run/reclosure 任务遇到 active blocker 时，
  先按治理同步删除或改成审计记录，再继续 fresh CUDA validation 和运行。
- 不要 fake validation；
- 不要改历史正式 run artifact；
- 不要绕过 One-Control；
- 不要让 tests/docs/guardrails 与 active code 不一致；
- 不要为了兼容旧版牺牲新架构收口。

## 9. 最终报告固定结构

最终报告必须包含：

### 0. 一句话结论

[目标对象] 当前状态属于：已完成 / 部分完成 / 尚未开始 / 失败 / 阻塞。

### 1. 标杆对象现行架构

列出标杆对象分层，并用文件路径 + 函数/类 + 行号支撑。

### 2. 目标对象现行架构

列出真实入口、真实运行路径、遗留点，并用文件路径 + 函数/类 + 行号支撑。

### 3. 两者逐层对比表

列格式：

架构层/组件 | 标杆对象状态 | 目标对象状态 | 判定结论

### 4. 最大技术债缺口

按优先级写：

P0：
P1：
P2：

每条包含：

problem | evidence path | impact | recommended next action

### 5. 最小重构边界

明确：

- 必改部分：
- 可复用部分：
- 不该动的部分：

### 6. 风险评估

说明：

- 如果现在直接进行下一阶段任务，有什么隐患；
- 如果先插小重构阶段，收益与代价是什么。

### 7. 最终建议

只能二选一：

- 直接推进下一步；
- 先插过渡重构阶段。

如果选择后者，必须一句话定义这个小阶段。

### 8. Validation Summary

表格：

validation item | command | result | evidence path | notes

### 9. Modified Files / Diff Summary

如果是修复任务，列出：

- 修改文件；
- 删除文件；
- 新增文件；
- 关键函数；
- 风险。

审计任务可写“无代码修改”。

### 10. Machine Verdict

嵌入 machine verdict JSON 内容。

## 10. Machine Verdict 基础字段

每个 verdict JSON 至少包含：

```json
{
  "verdict": "A/B/C/D",
  "p0_count": 0,
  "p1_count": 0,
  "p2_count": 0,
  "stage": "...",
  "task": 2,
  "code_modified": true,
  "formal_run_launched": true,
  "training_launched": true,
  "eval_launched": true,
  "notes": []
}
```

具体字段可按任务扩展。

## 11. 裁决标准

A：

目标完成，P0=0，关键验证 PASS。

B：

主体完成，但有非阻塞 P1/P2 或 runtime 未完成。

C：

存在 active P0、关键路径不符合架构、验证失败或证据不足。

D：

状态混乱，无法判断。

## 12. 后续提示词引用方式

后续任务不要复制本 skill 全文。

只写：

“输出与归档遵循 docs/skills/ODCR_AUDIT_DELIVERY_SKILL.md。”
