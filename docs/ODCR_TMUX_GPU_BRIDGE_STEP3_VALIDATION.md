# ODCR Tmux GPU Bridge Step3 Validation

Codex/admin shell is not the GPU shell. Step3 GPU runtime validation uses a
controlled tmux GPU bridge command sent to a user-created, already-entered,
uniquely validated GPU pane.

The manual user action is `odcr-enter-gpu <JOBID>` inside the existing tmux
session. Codex must not execute `odcr-enter-gpu`, `srun`, `sbatch`, or `scancel`, and it
does not create, kill, switch, or attach tmux sessions. The user enters the GPU
allocation manually inside the existing tmux session; the bridge only sends a
bridge-generated command file after fresh discover and fresh validate evidence
for the current pane. GPU use is allowed by default for repo-local validation,
probe, and bounded runtime; post-edit full is not a GPU prerequisite.

The Step3 bridge mode is `step3-startup-validation`. It is not formal, not
short-pilot, and not a parameter experiment. It does not launch full Step3,
does not start Step4/Step5/eval/rerank, does not write formal checkpoints, and
does not update `runs/step3/task2/latest.json`.

Expected sequence:

1. `discover`
2. `validate-only --strict`
3. `marker-probe`
4. `cuda-probe`
5. `step3-startup-validation`

Every run must fresh discover the current pane and fresh validate the live
CUDA environment. Old state files, old `AI_analysis` probe output, and a tmux
session that is still on an admin node are not evidence.

The validation writes only isolated evidence:

- `AI_analysis/06_probe_evidence/step3_tmux_gpu_bridge_startup_validation_closeout/`
- `runs/step3_validation/step3_tmux_gpu_bridge_startup_validation_closeout/`

A passing `step3-startup-validation` means the Step3 startup P0 contract has
been reproduced under the current 2-rank GPU runtime. It is still not a
training result or a paper metric. Only after this pass may the next handoff
request a formal task2 Step3 startup prompt.
