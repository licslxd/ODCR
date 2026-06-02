# ODCR Reusable Skills

后续 ODCR Codex 任务必须先读取：

- `docs/skills/ODCR_GPU_SKILL.md`
- `docs/skills/ODCR_AUDIT_DELIVERY_SKILL.md`
- `docs/skills/ODCR_CACHE_SKILL.md`
- `docs/skills/ODCR_TASK_ADAPTIVE_SAMPLER_CACHE_SKILL.md`
- `docs/skills/ODCR_PAPER_CHAT_CODEX_SKILL.md`

使用方式：

GPU 相关执行遵循 `docs/skills/ODCR_GPU_SKILL.md`。
输出与归档遵循 `docs/skills/ODCR_AUDIT_DELIVERY_SKILL.md`。
缓存相关设计遵循 `docs/skills/ODCR_CACHE_SKILL.md`。
采样、Gold/CF 分档、compact length、ablation 和缓存复用联动问题遵循
`docs/skills/ODCR_TASK_ADAPTIVE_SAMPLER_CACHE_SKILL.md`。
论文 Chat/Codex 分工遵循 `docs/skills/ODCR_PAPER_CHAT_CODEX_SKILL.md`。

不要在每个任务提示词里复制 skill 全文。
