# ODCR_TASK_ADAPTIVE_SAMPLER_CACHE_SKILL

## 1. 适用范围

本 skill 用于 ODCR 中涉及以下内容的审计、设计或修改：

- task-specific sampler / sample plan / Step5 pool selection；
- Gold High / Medium / Reject 保留、降权或剔除策略；
- CF high / medium / low_weighted / reject 分档权重与路由语义；
- ratio 配比、sample weight、route bypass、ablation candidate 命名；
- compact evidence 的 max length / truncation / token length 统计；
- sampled train table、token cache、control tensor cache、lineage 复用。

本 skill 是动态决策规则，不是固定 Task8 配方。每个 task 必须先看数据分布、
阶段语义、论文约束和缓存产物边界，再决定是否复用、降权、剔除或重建。

## 2. 核心原则

不要把“抑制主导”实现成“整类删除”。

- 高质量 Gold 默认应保留，可 cap、降权或做独立 ablation；只有存在泄漏、污染、
  明确任务不适配或 Chat/实验设计要求时，才允许整类删除。
- CF 默认应按可靠度分档，而不是全局低权重一刀切。优质 CF 是跨平台泛化约束，
  低质 CF 才是噪声控制对象。
- ratio、sample weight、Gold tier、CF tier、route policy 必须能单独开关或单独命名，
  不得用一个 candidate token 隐式绑定多个实验变量。
- 缓存 identity 只包含会改变缓存产物内容的字段；审计、运行、优化和论文解释字段
  进入 lineage，不得让小改权重导致无意义全量重算。

## 3. 每个 Task 必查证据

在下结论前，至少整理以下表格或等价摘要：

- task / domain regime：同平台、跨平台、弱跨平台、数据稀缺程度、stage 消费语义；
- Gold pool：target/aux 的 high、medium、reject 数量、比例、是否 route-compatible；
- CF pool：high、medium、low_weighted、reject 数量，`route_explainer`、
  `route_scorer`、`train_keep`、posterior sample weight、reliability 均值；
- effective mass：每类样本的 row count、平均权重、总权重质量，而不只看行数；
- length stats：p50/p90/p95/p99/max、超过 compact 长度的数量、真实截断数量；
- cache boundary：当前缓存对象到底包含 rows、row order、token ids、labels、
  control tensors、sample weights、generated text 还是 metrics。

## 4. 动态判定规则

### Gold tier

- High 非空且无污染证据：默认保留，必要时降权或设上限。
- Medium：通常作为常规主体样本，保留正常权重。
- Reject：默认剔除，除非任务明确做噪声鲁棒诊断且 candidate 名称显式标注。
- 如果要删除 High，必须同时提供 High-vs-kept 的消融计划，否则论文无法论证收益。

### CF tier

- High / Medium CF 如果可靠且 route-compatible，默认不应被全局低权重吞掉。
- CF 抽样必须先遵守 component 配比，再在 CF component 内按 tier 配比抽取；
  tier 分配不得反向改变 `target_gold / aux_gold / cf` 的总体比例。
- CF tier 的正式填充顺序是 High 优先、Medium 补足。若 High 不足，先把缺口转给
  Medium；若 High + Medium 仍不足，再决定是否允许少量 Low_weighted。
- Low_weighted CF 只能作为覆盖补充或显式诊断；进入正式候选时必须同时设置 cap
  与降权，记录触发原因、替代缺口、行数、平均权重、总 effective mass，并在
  candidate 名称或审计字段中标明 low_weighted fallback。
- Reject CF 默认不进训练；仅可用于诊断，不可伪装成正式 counterfactual evidence。
- route bypass 只能作为显式诊断候选，不能藏在“弱协议”里成为默认正式语义。

### Ratio 与 sample weight

- component ratio 是第一层采样语义：先确定 `target_gold / aux_gold / cf` 的样本
  质量预算，再在各 component 内部分配 tier。不得用 CF tier 短缺把 CF 整体压成
  接近零，除非候选明确声明为 target/gold-only 或 CF-disabled ablation。
- Step5 formal sampler 必须按总数执行三层预算，不得只靠固定 tier ratio：
  1. 先确定 `N = effective_samples_per_epoch`；
  2. 再按 component ratio 得到 `target_gold_N / aux_gold_N / cf_N`；
  3. Gold 在本 component 总量内优先 High，不够用 Medium 补；
  4. CF 在 `cf_N` 总 cap 内优先 High，不够补 Medium，High+Medium 仍不足时才允许
     Low_weighted fallback；
  5. Low_weighted fallback 必须同时记录 cap、实际行数、权重、未满足缺口和触发原因。
- 对 Task8 这类 High/Medium CF 稀缺任务，推荐先把 CF 总量设成小而真实的 cap
  （如 `N=250k` 时 `cf_N=12.5k`），然后让 High/Medium 全收、Low_weighted 只补缺口；
  不允许因为 High/Medium 少就把 CF ratio 压成接近 0。
- 固定配比关闭时，必须确认 sample weight 足以提供有效迭代步数。
- 固定配比与全局低权重同时削弱同一辅助源时，要计算双重削弱后的 effective mass。
- 配比策略、CF 权重、Gold 筛选、route policy 的 ablation 名称必须解耦。

### Step5 训练/评估接口一致性

- Step5 训练输入格式、训练 evidence 来源、训练目标必须与最终评估生成任务一致；
  如果最终评估是 no-reference history/evidence 输入，训练也必须使用同一类
  no-reference evidence，而不是训练时吃 Step4 当前行 EASD/HSS、评估时换成 history。
- 会改变 tokenized input、label、control text、sample rows 或 sample weights 的接口修改，
  必须触发 Step5 sample-plan/token cache rebuild；只改变 epoch、lr、loss weight、decode knob
  不应单独重建 eval token cache。

### Compact length

- 如果论文或模型声明使用 32-token compact evidence，默认把 32 作为主配置候选。
- 若 p99 显著超过 32 或截断损失可能影响任务，需要显式比较 32 vs 48/64/128，
  并同步修改论文叙述；不得一边写 32-token，一边无解释使用 128。
- length stats 只能从当前 source/processor/max length 对应的 manifest 或 data audit 读取；
  预处理逻辑改变后，旧长度统计必须视作 stale，除非 lineage 校验通过。

## 5. 缓存与复用规则

缓存规则遵循 `docs/skills/ODCR_CACHE_SKILL.md`，并额外注意：

- sampled train table cache identity 应包含会改变行集合、行顺序、样本权重、
  route/control 字段或 provenance 的 sample plan 语义。
- token cache 如果包含 `exp_sample_weight`、control tensors 或 label tensors，
  对应字段就是 content-affecting，必须进入 identity。
- 如果希望 weight-only sweep 不重建 token ids，应拆分为：
  token ids cache + mutable row/control/weight table，而不是复用 stale token cache。
- broad resolved config hash、run id、epoch、loss weight、runtime diagnostics 不得作为
  token cache identity 的唯一依据。

### RACER-C1 / Retrieval-first Step5 reset

当任务明确转向 `RACER-C1` 这类 retrieval-first 方法时，必须物理删除旧大模型生成器
正式代码；不得保留 FLAN/T5/LoRA baseline、历史兼容测试或隐藏 fallback。

- 新主线语义：train-only evidence pool -> metric-aligned contrastive retriever ->
  RCR-aware rerank -> top-1 evidence prediction / compact composer。
- evidence pool、pair manifest、embedding cache、prediction provenance 都是内容产物；
  task、source split、清洗版本、25-token 截断、embedding backbone、embed_dim、schema
  version 必须进入 cache identity。
- epoch、lr、batch、target GPU memory、runtime utilization、throughput 只进 lineage；
  这些调度/优化字段不得让同一份 evidence embedding cache 无意义重建。
- 训练日志必须默认写入 `runs/racer_c1/task*/<run>/meta`，至少包含 per-epoch loss、
  elapsed time、pairs/sec、tokens/sec、GPU/CPU utilization、GPU memory、dataloader/CPU
  bottleneck 判断；诊断文件写入 `diagnostics/`。
- 如果 GPU 显存没有接近目标值（例如每卡 35GB），报告必须解释是 CPU/embedding
  cache/dataloader 限制、batch 太小、序列太短、模型太小，还是负样本挖掘耗时导致，
  不能简单说“没有吃满显存”。
- 发现旧 Step5 大模型/多候选 rerank/LoRA 路径仍在 active code/tests/config 里被当成
  可运行主线时，先停止当前实现并物理清理；不允许用包装、alias、baseline-only 或
  silent fallback 掩盖旧逻辑。

## 6. 常见红旗

- sampler 名称里出现 `MEDIUM_ONLY`，但 High pool 非空且没有污染证据；
- CF candidate 把所有 CF 都压成 low_weighted，或用 route bypass 人为放大低质 CF；
- CF 总比例被 tier 短缺、route filter 或 fallback 逻辑压到接近零，但 candidate
  名称仍声称是 CF/RCR 主线；
- High CF 不足时直接跳到 low_weighted，而没有先用 Medium 补足、记录缺口和
  effective mass；
- `STEP*_RATIO_0` 与全局低权重同时作用在稀缺辅助数据上；
- 一个 candidate token 同时绑定 ratio、Gold filter、CF weight、route policy；
- cache identity 排除了缓存产物实际包含的 `sample_weight_hint` / `exp_sample_weight`；
- max length 远大于 compact encoder 论文定义，且没有 32/48/64 对照解释；
- data audit 的长度统计没有 source/processor/hash 校验锁。

## 7. 输出要求

相关任务最终至少交付：

- component decision table：Gold High/Medium、CF High/Medium/Low/Reject 分别
  keep / downweight / drop / diagnostic-only；
- effective mass table：行数、平均权重、总权重质量；
- CF fallback table：High 请求/可用/实际、Medium 请求/补缺/实际、Low_weighted
  cap/实际/权重、未满足缺口、是否 route-compatible；
- ablation matrix：每个候选只改变一个主要变量；
- cache reuse decision：哪些 cache 可复用，哪些必须重建，理由是什么；
- paper risk note：若配置和论文 32-token compact / weak cross-platform CF 叙述冲突，
  必须标明是正式候选、诊断候选还是待 Chat 决策候选；
- rerun decision：是否需要 preprocess、Step3、Step4、Step5、eval 或 rerank rerun。

## 8. 禁止事项

- 不得默认删除全部 High Gold。
- 不得默认把全部 CF 一刀切低权重。
- 不得把低质量 CF 的 route bypass 写成正式可靠性门控证据。
- 不得把三个以上实验变量绑在同一个不可拆 candidate 名称中。
- 不得用 stale length stats 支撑新的 max length / truncation 结论。
- 不得用缓存复用掩盖 sample weight、control tensor 或 schema 变化。
- 不得运行训练、eval、rerank 或 GPU formal work，除非用户另行明确授权。
