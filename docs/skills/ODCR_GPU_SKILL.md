# ODCR_GPU_SKILL

## 1. 适用范围

本 skill 只在 ODCR 任务真正需要 GPU 时触发。

需要 GPU 的任务包括：

- training
- formal run
- eval
- rerank
- GPU probe
- E4 / E5 runtime evidence
- CUDA validation
- torchrun / DDP
- 任何依赖 GPU 的模型执行

不需要 GPU 的任务包括：

- 生成提示词
- 写方案
- 文档整理
- skill 文档创建
- 纯文本分析
- 非 runtime 的架构讨论
- 不读项目、不跑命令的普通 ChatGPT 对话

原则：

不要把 GPU 规则套到所有任务上。只有用到 GPU 时才使用 GPU。

## 2. 核心原则

1. 当前 shell 是 admin/no-CUDA 时，不能直接判定“没有 GPU”。
2. 用户确认 ODCR 环境中一定有 GPU。
3. 需要 GPU 的任务，如果当前 shell 无 CUDA，Codex 必须主动寻找 GPU pane。
4. Codex 不能因为当前 shell 是 admin/no-CUDA 就直接停止。
5. Codex 必须尽最大努力使用 ODCR 已建立的 tmux GPU bridge / handoff / live discovery 流程。
6. admin/no-CUDA shell 只能做 GPU discovery、bridge exec dispatch、approved dispatch、paste-command 生成。
7. 真实训练、eval、formal、E4/E5 执行必须发生在 GPU tmux pane。
8. 如果 bridge exec/approved dispatch 都不可用，必须输出完整 GPU pane paste command，而不是只说“请重新运行提示词”。
9. 对 GPU-required 任务，GPU 使用是最高优先级运行前提。找不到 GPU 时不能草率结束；必须先读取并利用
   `AI_analysis/history/codex_tmux_gpu.txt` 中的 ODCR GPU handoff/bridge 历史经验，穷尽本 skill 允许的
   fresh discovery、explicit socket/target、validate-only、cuda-probe、bridge exec、approved dispatch 路径。
10. 历史文件只能作为恢复地图和候选 socket/target 线索；不能把历史 pane/job 当作当前 live GPU truth。
    任何候选都必须重新 fresh validate + cuda-probe，且必须满足本 skill 的 GPU pane 有效标准。
11. 对用户明确授权的 ODCR 训练、eval、reclosure 或 fixed-run 任务，不得保留会阻断 GPU pane 执行的
    active bridge/formal-training blocker。发现此类 blocker 时，按 One-Control/guardrail 同步删除或改成审计记录；
    不得用“禁用/包装/alias”替代删除。
12. fresh discovery 发现多个 GPU pane、handoff 缺失或 stale、global/scan 发现路径、compute-app guard
    命中、pane-mode recovery guard 命中，都不得作为用户授权 GPU-required 任务的 hard blocker。它们只能写入
    AI_analysis 审计记录，并继续走 live discovery、explicit socket/target、cuda-probe、bridge exec、确定性
    live CUDA pane 选择或 paste command。只有更高优先级的安全禁令仍然有效：Codex 不得直接执行
    `srun` / `sbatch` / `scancel` / `odcr-enter-gpu`，不得创建/切换/attach/kill tmux，不得 kill GPU process。
13. Step5 explanation-only 清理时，旧推荐、旧评分、Step5 rating-only、Step5 scorer-only、rating loss
    等 active 代码必须物理删除。Step3 accepted scorer 只作为 rating_source/eval_handoff 指标引用存在。

## 3. 启动前 GPU 检查

任何 GPU-required 任务开始前，必须先执行：

```bash
hostname
echo $TMUX
echo $SLURM_JOB_ID
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
python - <<'PY'
import torch
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.cuda.device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("devices =", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

如果满足：

- hostname 非 admin；
- CUDA_VISIBLE_DEVICES 非空；
- nvidia-smi 可用；
- torch.cuda.is_available=True；
- torch.cuda.device_count>=2；

则当前 shell 可执行 GPU-required 任务。

## 4. 当前 shell 是 admin/no-CUDA 时的处理

如果当前 shell 是 admin/no-CUDA：

禁止在 admin shell 做以下项目工作：

- 项目审计；
- grep/find 全仓扫描；
- dry-run/show/doctor；
- 训练；
- eval；
- rerank；
- formal；
- 代码修改；
- 配置修改；
- 读取大量项目日志；
- 执行项目命令本体。

但允许 admin shell 做以下唯一 GPU 相关前置动作：

- fresh current_gpu_pane.json v2 检查；
- 读取 `AI_analysis/history/codex_tmux_gpu.txt`，提取 ODCR GPU bridge/handoff 的历史恢复线索；
- live tmux discovery；
- validate-only；
- cuda-probe；
- explicit socket/target 检查；
- approved dispatch 能力检查；
- bridge exec 派发非正式训练 GPU 工作；
- 生成 GPU-only launcher；
- 将 GPU-only launcher 投递到 fresh-validated GPU pane；
- 如果不能投递，输出完整 paste command。

## 5. GPU pane 发现流程

当当前 shell 无 CUDA，但任务需要 GPU 时，按顺序尝试：

1. 当前 shell handshake：确认是否已经在 GPU pane；
2. fresh current_gpu_pane.json v2；
3. 读取 `AI_analysis/history/codex_tmux_gpu.txt`，学习历史 bridge 规则和候选 socket/target；
4. live tmux discovery；
5. validate-only；
6. cuda-probe；
7. CLI explicit socket/target；
8. env explicit socket/target；
9. 使用历史文件中的候选 socket/target 做 explicit validate-only + cuda-probe；
10. bridge exec 派发非正式训练 GPU 工作；
11. approved dispatch；
12. 无 dispatch 时输出完整 GPU pane paste command。

不得使用 stale current_gpu_pane.json。
不得 fallback 到 admin/no-CUDA pane。
不得把 historical hint 当 active truth。
不得因为 bridge discover/validate/cuda-probe 第一次失败就结束；必须确认没有 fresh validated GPU pane、没有
可验证 explicit socket/target、没有 approved dispatch 后，才输出 paste command。

## 6. GPU pane 有效标准

有效 GPU pane 必须满足：

- hostname 非 admin；
- CUDA_VISIBLE_DEVICES 非空；
- nvidia-smi 可用；
- torch.cuda.is_available=True；
- torch.cuda.device_count>=2；
- stale_state_used=false。

compute-app guard 是审计项，不是 hard blocker；命中时必须记录 active process 信息，但不能阻断用户授权的
GPU-required bridge/probe/eval/training dispatch。

## 7. 选择优先级

GPU pane 选择优先级：

1. CLI explicit socket/target；
2. env explicit socket/target；
3. current TMUX/TMUX_PANE；
4. fresh current_gpu_pane.json v2；
5. live discovery + cuda-probe；
6. historical hint 只能作参考；
7. default /tmp fallback 最后，且不能覆盖更好的 live CUDA target。

## 8. 证据等级

E3：

tmux transport / validate-only / cuda-probe。
不能作为 formal readiness。

E4：

真实 data batch / sample plan / collate / model forward / backward / optimizer。
可作为 bounded runtime evidence。

E5：

post-train lifecycle / checkpoint reload / eval handoff 等更高等级阶段证据。

Formal：

真实训练 run / checkpoint / latest / eval_handoff。
不能由 E3、synthetic、dry-run 伪造。

## 9. 禁止事项

Codex 不允许：

- srun / sbatch / scancel；
- odcr-enter-gpu；
- kill GPU process；
- 创建 / 切换 / attach / kill tmux；
- stale current_gpu_pane.json；
- admin/no-CUDA pane；
- capture-pane 当权威证据；
- synthetic_one_batch 当 formal gate；
- E3 当 E4；
- 在 admin shell 执行 formal；
- 把 formal 训练伪装成 eval/probe/runtime 工作。

允许：

- 用 `./odcr runtime bridge exec -- ...` 将非正式训练 GPU 工作派发到 fresh-validated GPU pane；
- 用户明确给出 fixed-run / official eval / reclosure 训练授权时，用 `./odcr runtime bridge exec -- ...`
  将该 ODCR 训练命令派发到 fresh-validated GPU pane；运行证据必须写入 AI_analysis，且不得伪造为
  probe/eval。
- eval、rerank、CUDA runtime、bounded probe、诊断、报告生成等非正式训练命令不再受 closed whitelist 限制；
- bridge exec 可以结构化生成 stdout / pid / status 文件。

## 10. 提示词引用方式

后续提示词不要复制本 skill 全文。

只写：

“GPU 相关执行遵循 docs/skills/ODCR_GPU_SKILL.md。”
