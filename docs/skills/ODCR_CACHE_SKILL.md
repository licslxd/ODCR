# ODCR_CACHE_SKILL

## 1. 适用范围

本 skill 适用于 ODCR 中任何会读取、写入、复用、迁移或审计缓存的任务，包括：

- preprocess cache；
- tokenizer / HF dataset cache；
- Step4 / Step5 export loader cache；
- eval / rerank cache；
- cache manifest、lineage、fingerprint、reuse decision；
- 因 cache miss / cache hit 异常导致的训练、评估、rerank 延迟或结果差异分析。

后续提示词只需要写：

“缓存相关设计遵循 `docs/skills/ODCR_CACHE_SKILL.md`。”

不要在每个任务里复制本 skill 全文。

## 2. 核心原则

缓存 key 只能包含会改变缓存产物内容的因素。

不会改变缓存产物内容的因素必须写入 manifest / lineage / audit metadata，
但不得作为 cache invalidator 导致重建。

换句话说：

- 影响 cached artifact bytes / rows / tensors / token ids 的因素：进 cache key。
- 只影响训练目标、优化、解码、运行环境、调度、日志或审计解释的因素：进 lineage，不进 cache key。

任何新缓存或缓存修复都必须把 `cache_identity` 和 `lineage_metadata`
分开设计。不要用一个 broad resolved-config hash 同时承担“是否可复用”和“审计记录”的职责。

## 3. Token Cache Identity 规则

对于 Step5 tokenizer / HF dataset cache，只有以下因素应进入 token cache identity：

- 源数据内容：CSV/parquet/source artifact 的路径角色、内容 hash、行顺序、行选择结果；
- split / task / domain / head：仅当它们改变要 tokenized 的行或文本字段时；
- tokenizer artifact：tokenizer 文件、special tokens、vocab、tokenizer config 的 fingerprint；
- Processor / input formatting：会改变输入文本、label 文本、control text、prompt 模板的版本；
- max length / truncation：会改变 `input_ids`、`labels`、attention mask 或 control tensor 的长度/内容；
- required fields / schema version：会改变 Processor 读取字段、字段含义、缺失策略或 tensor 输出结构；
- eval control contract：仅当它改变 official valid/test 的 tokenized 输入、label 或 control packet；
- cache schema / producer code version：仅当 producer 改变缓存产物格式或 tensor 语义；
- sampling or sample plan：仅当缓存对象本身是 sampled train plan，且 sample plan 改变行集合、行顺序或 epoch 展开。

训练集 cache 和 official valid/test cache 不能混为一谈：

- train cache 可以包含 sample plan identity，因为 sample plan 改变训练行集合。
- valid/test official eval cache 不应因为训练 sampler plan、训练 epoch、loss 权重或 run lifecycle 改变而重建，
  只要 valid/test 源数据、Processor、tokenizer、max length 和 eval-control tokenization 语义不变。

## 4. 不应触发 Token Cache 重建的因素

以下因素默认不得进入 token cache identity，除非代码证据证明它们改变 tokenized artifact 内容：

- loss 权重、aux loss 开关、anti-collapse 权重、terminal clean 权重；
- learning rate、optimizer、warmup、epoch 数、early stopping、checkpoint selection；
- batch size、DDP world size、num workers、num_proc、pin_memory、prefetch；
- GPU 型号、hostname、SLURM_JOB_ID、PID、CUDA_VISIBLE_DEVICES、runtime timing；
- run_id、output directory、train-only / official-eval lifecycle；
- checkpoint path、checkpoint hash、model weights；
- decode / generation 参数，如 temperature、top_p、top_k、min_new_tokens、no_repeat_ngram_size、
  repetition_penalty、bad_words_ids，除非缓存的是生成结果而不是 tokenized inputs；
- broad `resolved_config_hash`、runtime diagnostics hash、training semantic hash；
- train sampler plan hash 对 official valid/test token cache 的影响。

这些因素仍然可以、也应该写入 lineage / manifest / source_table / audit report，
用于解释“这个 run 为什么这样跑”，但不能让同一份 tokenized valid/test 数据重复构建。

## 5. Manifest 与 Lineage

每个可复用缓存必须至少有两组元数据：

1. `cache_identity`

   只记录会改变缓存产物内容的字段。consumer 用它判断 hit/miss。

2. `lineage_metadata`

   记录完整上下文，包括 resolved config、run id、checkpoint、training semantics、
   runtime diagnostics、sampler plan、loss/generation knobs、producer version、source table 等。
   consumer 用它做审计和解释，不用它决定 token cache 是否必须重建。

如果现有实现只有一个 broad fingerprint，修复时必须迁移为两层，或明确把 broad 字段移出
cache invalidation gate。

## 6. Mismatch Policy

cache identity mismatch：

- 必须 fail-fast 或 rebuild；
- 必须写明具体字段；
- 不得 silent fallback；
- 不得硬链接、复制、重命名旧 cache 来绕过 manifest 校验。

lineage metadata mismatch：

- 不应触发 token cache rebuild；
- 必须在 manifest / reuse decision / audit report 中记录；
- 只有当 mismatch 证明会改变 cached artifact 内容时，才升级为 identity mismatch。

缺失 manifest、缺失 completed marker、schema 不兼容、source artifact hash 不匹配时，必须拒绝复用。

## 7. 修复缓存问题时的执行顺序

1. 先确认 cache 对象是什么：tokenized dataset、sampled train plan、export loader cache、generated output、metrics cache。
2. 列出 cached artifact 的真实内容边界：rows、columns、token ids、labels、control tensors、metrics、生成文本等。
3. 对每个 fingerprint 字段分类：
   - content-affecting -> cache identity；
   - audit-only -> lineage metadata；
   - obsolete/retired -> 删除或 retired-fail-fast。
4. 用现有 run 证据验证同源数据是否被错误 miss。
5. 修改实现时同步更新 manifest schema、reuse decision、tests、guardrail 或 docs。
6. 运行 narrow post-edit validation。
7. 说明哪些旧 cache 可继续复用，哪些必须重建。

## 8. 禁止事项

- 不得把 broad resolved config hash 当作唯一 token cache key。
- 不得因为 loss/generation/training-only 参数变化重建 valid/test token cache。
- 不得把 runtime timing、GPU 状态、hostname、PID、num_proc 放进 token cache identity。
- 不得用 symlink/copy 的方式伪造 cache hit。
- 不得让 `AI_analysis` 成为缓存 payload 存储目录；缓存 payload 属于 `cache/`。
- 不得为了复用缓存而接受缺失、stale、schema 不兼容或 source hash 不匹配的缓存。

## 9. 提示词引用方式

后续提示词不要复制本 skill 全文。

只写：

“缓存相关设计遵循 `docs/skills/ODCR_CACHE_SKILL.md`。”
