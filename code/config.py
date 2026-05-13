import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from dataclasses import asdict, dataclass, field, replace
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
)

from cpu_utils import effective_cpu_count

# 执行层：DDP_NPROC / torchrun --nproc_per_node 仅在 shell 解析；Python 以 WORLD_SIZE 为准。
# num_proc 为 CPU 侧（如 datasets.map）并行度，属 resolver-injected hardware 层，勿与 DDP_NPROC 混淆。详见 docs/ODCR_Scripts_and_Runtime_Guide.md。

# import 后由 train_logging.flush_preset_load_events 写入训练日志（摘要侧）
PRESET_LOAD_EVENTS: List[str] = []


def record_preset_event(msg: str) -> None:
    PRESET_LOAD_EVENTS.append(msg)


# ---------------------------------------------------------------------------
# 类型与轻量配置对象（供静态检查与逐步向「入口 resolve、后续只读」演进）
# ---------------------------------------------------------------------------


class TaskConfig(TypedDict):
    """单任务默认表；旧 adv/eta 语义已退役，不再作为任务字段。"""

    auxiliary: str
    target: str
    lr: Union[int, float]
    coef: Union[int, float]
    scenario: str
    direction: str


class HardwarePresetRow(TypedDict, total=False):
    """命名 hardware 预设（CPU / DataLoader / DDP 布局等）；与训练 TRAINING_PRESETS 独立。"""

    max_parallel_cpu: int
    num_proc: int
    max_num_proc: int
    reserved_cpu: int
    tokenization_num_proc: int
    ddp_world_size: int
    dataloader_num_workers_train: int
    dataloader_num_workers_valid: int
    dataloader_num_workers_test: int
    dataloader_prefetch_factor_train: int
    dataloader_prefetch_factor_valid: int
    dataloader_prefetch_factor_test: int
    dataloader_workers_train_per_rank_cap: int
    omp_num_threads: int
    mkl_num_threads: int
    tokenizers_parallelism: bool
    pin_memory: bool
    persistent_workers: bool
    non_blocking_h2d: bool


class TrainingPresetRow(TypedDict, total=False):
    """命名预设中允许出现的字段（全局一条或 per-task 子 dict）；与 _TRAINING_PRESET_ALLOWED_KEYS 对齐。"""

    train_batch_size: int
    train_label_max_length: int
    epochs: int
    full_bleu_eval: Dict[str, Any]
    min_lr_ratio: float
    lr: float
    coef: float
    per_device_train_batch_size: int
    per_gpu_batch_size: int
    train_dynamic_padding: bool
    loss_weight_repeat_ul: float
    loss_weight_terminal_clean: float
    terminal_clean_span: int
    full_bleu_decode_strategy: str
    batch_diversity_use_ema: bool
    batch_diversity_ema_decay: float
    batch_diversity_min_valid_tokens: int
    batch_diversity_loss_clamp_abs: float
    lambda_ortho: float
    lambda_ortho_xcov: float
    lambda_ortho_cos: float
    lambda_ortho_step5: float


@dataclass(frozen=True)
class BaseTrainingDefaults:
    """
    训练相关**代码默认**的单一来源（不含 task 表、不含 preset、不在 import 时读 ENV）。

    解析链目标形态：BASE_TRAINING_DEFAULTS → One-Control YAML / CLI → 解析结果。
    """

    epochs: int = 50
    train_batch_size: int = 2048
    min_lr_ratio: float = 0.1
    train_min_epochs: int = 8
    train_early_stop_patience: int = 6
    train_bleu4_max_samples: int = 512
    lr_scheduler: str = "warmup_cosine"
    warmup_epochs: float = 1.0
    # eval 全局 batch 的代码默认；torchrun 子进程内实际值来自父进程写入的 effective training_row（见 build_resolved_training_config）
    eval_batch_size: int = 2560
    # 无任务 lr 时 learning_rate resolve 链末级 fallback（在 build_resolved_training_config 内使用）
    initial_learning_rate: float = 1e-3
    # 训练标签（teacher forcing）最大 token 长度；与 decode 的 max_explanation_length 解耦
    train_label_max_length: int = 64


BASE_TRAINING_DEFAULTS = BaseTrainingDefaults()
DEFAULT_TRAINING_CONFIG = BASE_TRAINING_DEFAULTS
"""与 BASE_TRAINING_DEFAULTS 同义别名，便于语义上称「默认训练配置」。"""

# 模块级便捷常量：与 ``from config import train_batch_size`` / ``epochs`` 等一致（值均来自 BASE_TRAINING_DEFAULTS）
train_batch_size = BASE_TRAINING_DEFAULTS.train_batch_size
epochs = BASE_TRAINING_DEFAULTS.epochs


T = TypeVar("T")


def _preset_int_min(raw: Any, minimum: int) -> int:
    return max(minimum, int(raw))


def _coerce_task_param_numeric(v: Any) -> Union[int, float]:
    """与 task_configs 中 coef 等为整数时的 format 输出兼容：整数值用 int，否则 float。"""
    fv = float(v)
    if math.isfinite(fv) and fv.is_integer() and abs(fv) <= 2**53:
        return int(fv)
    return fv


# ---------------------------------------------------------------------------
# One-Control support tables: no sidecar YAML or loose preset files are read.
# ---------------------------------------------------------------------------


_TASK_ROW_KEYS: FrozenSet[str] = frozenset({"auxiliary", "target", "source", "lr", "coef", "scenario", "direction"})


def _normalize_task_row_yaml(v: Any, *, ctx: str) -> TaskConfig:
    if not isinstance(v, dict):
        raise TypeError(f"{ctx} 须为 dict，当前为 {type(v).__name__}")
    unk = set(v.keys()) - _TASK_ROW_KEYS
    if unk:
        raise ValueError(f"{ctx} 含有未知字段 {sorted(unk)}")
    source_present = "auxiliary" in v or "source" in v
    required = {"target"}
    miss = required - set(v.keys())
    if not source_present:
        miss.add("source")
    if miss:
        raise ValueError(f"{ctx} 缺少字段 {sorted(miss)}")
    return {
        "auxiliary": str(v.get("auxiliary", v.get("source"))),
        "target": str(v["target"]),
        "lr": _coerce_task_param_numeric(v.get("lr", 1e-3)),
        "coef": _coerce_task_param_numeric(v.get("coef", 0.0)),
        "scenario": str(v.get("scenario", "legacy_scenario")),
        "direction": str(v.get("direction", "unspecified")),
    }


def _load_training_presets_from_yaml_required() -> Dict[str, Any]:
    """
    Empty compatibility table for internal helpers.

    The user control plane never reads scattered training preset files. The
    torchrun children receive their effective training row through
    ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON, produced by configs/odcr.yaml.
    """
    return {}


def _normalize_cuda_visible_devices_yaml(val: Any) -> Optional[str]:
    """hardware preset 中的 cuda_visible_devices：规范化逗号分隔设备列表；空则 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("null", "none", "~"):
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    return ",".join(parts)


def _coerce_hardware_yaml_value(key: str, val: Any) -> Any:
    """hardware YAML 标量：整数字段用 int；tokenizers_parallelism 用 bool。"""
    k = str(key)
    if k == "tokenizers_parallelism":
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("true", "1", "yes", "on")
    if k in ("pin_memory", "persistent_workers", "non_blocking_h2d"):
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("true", "1", "yes", "on")
    if val is None:
        raise ValueError(f"hardware 字段 {k!r} 不可为 null")
    if isinstance(val, bool):
        raise TypeError(f"hardware 字段 {k!r} 为 bool，仅 tokenizers_parallelism 允许布尔类型")
    return int(val)


@dataclass(frozen=True)
class FullBleuEvalResolved:
    """由 training_row.full_bleu_eval 解析得到的唯一 full BLEU 调度语义。"""

    mode: str  # "off" | "interval"
    every_epochs: Optional[int]
    enabled: bool

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "every_epochs": self.every_epochs, "enabled": bool(self.enabled)}


# 键名刻意拆写，避免仓库内 grep 旧符号时与本拒绝表误报为「仍在用」
_LEGACY_FULL_BLEU_KEYS: FrozenSet[str] = frozenset(
    (
        f"{'full_eval'}_{'every_epochs'}",
        f"{'full_eval'}_{'phased'}",
        f"{'full_bleu_eval'}_{'every_epochs'}",
    )
)


def _reject_legacy_full_bleu_keys(row: Mapping[str, Any]) -> None:
    bad = sorted(k for k in _LEGACY_FULL_BLEU_KEYS if k in row)
    if bad:
        raise ValueError(
            "training_row 含已废弃字段 "
            f"{bad}；请删除并改用唯一块 full_bleu_eval: "
            "{{ mode: off | interval, every_epochs: <int>（仅 interval 且 >0） }}。"
        )


def parse_full_bleu_eval_block(block: Any, *, ctx: str = "full_bleu_eval") -> FullBleuEvalResolved:
    if not isinstance(block, dict):
        raise TypeError(f"{ctx} 须为 dict，当前为 {type(block).__name__}")
    unk = set(block.keys()) - {"mode", "every_epochs"}
    if unk:
        raise ValueError(f"{ctx} 含有未知字段 {sorted(unk)}")
    raw_mode = block.get("mode", "")
    # PyYAML 1.1 会把 off/on 解析成 bool；此处与字符串 mode 等价处理
    if isinstance(raw_mode, bool):
        if raw_mode is False:
            mode = "off"
        else:
            raise ValueError(f"{ctx}.mode 为布尔真值，非法；请使用字符串 off 或 interval")
    else:
        mode = str(raw_mode).strip().lower()
    if mode not in ("off", "interval"):
        raise ValueError(f"{ctx}.mode 须为 off 或 interval，当前为 {block.get('mode')!r}")
    if mode == "off":
        if "every_epochs" in block and block["every_epochs"] is not None:
            raise ValueError(f"{ctx}: mode=off 时不应设置 every_epochs")
        return FullBleuEvalResolved(mode="off", every_epochs=None, enabled=False)
    raw_ee = block.get("every_epochs", None)
    if raw_ee is None:
        raise ValueError(f"{ctx}: mode=interval 时必须提供 every_epochs（正整数）")
    try:
        ee = int(raw_ee)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{ctx}.every_epochs 须为整数，当前为 {raw_ee!r}") from e
    if ee <= 0:
        raise ValueError(f"{ctx}.every_epochs 须 > 0，当前为 {ee}")
    return FullBleuEvalResolved(mode="interval", every_epochs=ee, enabled=True)


def resolve_full_bleu_eval_from_training_row(row: Mapping[str, Any]) -> FullBleuEvalResolved:
    """仅从 training_row 读取 full_bleu_eval；缺失时默认 off。"""
    _reject_legacy_full_bleu_keys(row)
    if "full_bleu_eval" not in row:
        return FullBleuEvalResolved(mode="off", every_epochs=None, enabled=False)
    return parse_full_bleu_eval_block(row["full_bleu_eval"], ctx="training_row.full_bleu_eval")


def should_run_full_bleu_eval_epoch(
    epoch_1_based: int,
    schedule: FullBleuEvalResolved,
) -> bool:
    """epoch_1_based 为从 1 开始的 epoch 序号。"""
    if not schedule.enabled or schedule.mode != "interval":
        return False
    ee = schedule.every_epochs
    assert ee is not None and ee > 0
    return epoch_1_based % ee == 0


def format_full_bleu_eval_resolved_log_line(schedule: FullBleuEvalResolved) -> str:
    return (
        f"[full_bleu_eval] mode={schedule.mode} every_epochs={schedule.every_epochs} "
        f"enabled={1 if schedule.enabled else 0}"
    )


def format_full_bleu_eval_epoch_decision_log_line(epoch_1_based: int, should_run: bool) -> str:
    return f"[full_bleu_eval] epoch={epoch_1_based} should_run={1 if should_run else 0}"


def parse_full_bleu_decode_strategy(v: Any, *, ctx: str = "full_bleu_decode_strategy") -> str:
    """训练期 full BLEU 监控用的解码策略：greedy=监控独立 greedy；inherit=与主 decode_strategy 一致。"""
    s = str(v).strip().lower()
    if s in ("greedy", "inherit"):
        return s
    raise ValueError(f"{ctx} 必须为 greedy 或 inherit，当前为 {v!r}")


def parse_checkpoint_selection_mode(v: Any, *, ctx: str = "checkpoint_selection_mode") -> str:
    """canonical downstream checkpoint mode: guarded composite only."""
    s = str(v).strip().lower()
    if s == "guarded_composite":
        return s
    if s == "valid_loss_only":
        raise ValueError(f"{ctx}=valid_loss_only is retired; use guarded_composite.")
    raise ValueError(f"{ctx} 须为 guarded_composite，当前为 {v!r}")


def parse_train_mode(v: Any, *, ctx: str = "train_mode") -> str:
    """Step5 文本侧：lora=仅训练 LoRA 旁路；full=全参数（禁止作为 Step5B 默认）。"""
    s = str(v).strip().lower()
    if s in ("lora", "full"):
        return s
    raise ValueError(f"{ctx} 须为 lora 或 full，当前为 {v!r}")


def parse_train_precision(v: Any, *, ctx: str = "train_precision") -> str:
    """训练期精度语义（与 ``ODCR_RUNTIME_PRECISION_MODE`` 编排对齐；manifest 审计用）。"""
    s = str(v).strip().lower()
    if s in ("bf16", "fp16", "fp32"):
        return s
    raise ValueError(f"{ctx} 须为 bf16、fp16 或 fp32，当前为 {v!r}")


def parse_batch_diversity_mode(v: Any, *, ctx: str = "batch_diversity_mode") -> str:
    s = str(v).strip().lower()
    if s in ("mean_prob_neg_entropy",):
        return s
    raise ValueError(f"{ctx} 须为 mean_prob_neg_entropy，当前为 {v!r}")


def parse_batch_diversity_ema_init_mode(v: Any, *, ctx: str = "batch_diversity_ema_init_mode") -> str:
    s = str(v).strip().lower()
    if s in ("uniform", "zeros"):
        return s
    raise ValueError(f"{ctx} 须为 uniform 或 zeros，当前为 {v!r}")


def apply_ddp_fast_torch_backends() -> bool:
    """
    ODCR_DDP_FAST 未禁用时在 CUDA 上启用 TF32、cudnn.benchmark（加速，数值可能因硬件略有差异）。
    须在 CUDA 可用且尽量在 set_device 之后调用。
    """
    import torch

    v = os.environ.get("ODCR_DDP_FAST", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    allow_tf32 = os.environ.get("ODCR_RUNTIME_ALLOW_TF32", "1").strip().lower()
    if allow_tf32 in ("0", "false", "no", "off"):
        return False
    if not torch.cuda.is_available():
        return False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return True


# ---------------------------------------------------------------------------
# 全局默认（模块级常量）
# ---------------------------------------------------------------------------

# 全局配置（预处理 GPU 段嵌入）
# 预处理嵌入批次默认值；公开调整入口走 ./odcr / configs/odcr.yaml，辅脚本只作内部传输。
embed_batch_size = 512  # compute_embeddings 默认嵌入批次大小；A100 40G 主线先保守使用 512，稳定后再手动上探


def get_odcr_embed_dim() -> int:
    """
    双通道 profile 最后一维 / ``FinalTrainingConfig.emsize`` 的 One-Control 期望。

    活跃来源是 ``configs/odcr.yaml: env.embed_dim``。torchrun 子进程只接受父进程
    注入的 ``ODCR_RESOLVED_EMBED_DIM`` 或 effective payload 中的 runtime_roots；旧
    ``ODCR_EMBED_DIM`` 仅允许与解析值一致，不能作为用户侧覆盖来源。
    """
    raw_resolved = (os.environ.get("ODCR_RESOLVED_EMBED_DIM") or "").strip()
    if raw_resolved:
        v = int(raw_resolved)
    else:
        v = 0
        payload_raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
        if payload_raw:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError:
                payload = {}
            roots = payload.get("runtime_roots") if isinstance(payload, dict) else None
            if isinstance(roots, dict) and str(roots.get("embed_dim") or "").strip():
                v = int(roots["embed_dim"])
        if v <= 0:
            cfg_path = Path(os.environ.get("ODCR_ROOT") or Path(__file__).resolve().parent.parent) / "configs" / "odcr.yaml"
            try:
                import yaml

                raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                env_cfg = raw_cfg.get("env") if isinstance(raw_cfg, dict) else None
                v = int(env_cfg.get("embed_dim")) if isinstance(env_cfg, dict) else 0
            except Exception as exc:
                raise RuntimeError(
                    "无法解析 One-Control env.embed_dim；请通过 ./odcr 或 python code/odcr.py 启动。"
                ) from exc
    legacy = (os.environ.get("ODCR_EMBED_DIM") or "").strip()
    if legacy and int(legacy) != int(v):
        raise ValueError(
            f"ODCR_EMBED_DIM={int(legacy)} conflicts with One-Control env.embed_dim={int(v)}; "
            "legacy env cannot override configs/odcr.yaml."
        )
    if v <= 0:
        raise ValueError("configs/odcr.yaml env.embed_dim 须为正整数")
    return v


# 全局配置（CPU 并行）
# 与作业实际可用核数对齐：优先 sched_getaffinity；不在 import 时冻结，见 _get_num_cpu() / get_num_proc()
def _get_num_cpu() -> int:
    return int(effective_cpu_count() or 8)


def _resolve_max_parallel_cpu_cli(max_parallel_cli: Optional[int] = None) -> int:
    """resolver-injected hardware profile only; child CLI/env fallbacks are forbidden."""
    _ = max_parallel_cli
    rp = _active_hardware_preset_slice()
    if rp and "max_parallel_cpu" in rp:
        return max(1, int(rp["max_parallel_cpu"]))
    raise RuntimeError("ODCR_HARDWARE_PROFILE_JSON 缺少 max_parallel_cpu；hardware 值必须由 resolver 注入。")


def _get_max_parallel_cpu() -> int:
    return _resolve_max_parallel_cpu_cli(None)

# 全局配置（Step 3/4/5 训练与推理）
# 代码默认已收敛至 BASE_TRAINING_DEFAULTS；本段仅说明语义与覆盖方式。
#
# train_batch_size / per_gpu_batch_size / epochs：所有 active train stages use no-accum semantics.
#   G = per_gpu_batch_size × world_size。
# eval 全局 batch：torchrun 子进程由 build_resolved_training_config 从 effective payload 读取；get_eval_batch_size() 仅服务
# 未走 odcr torchrun 的辅脚本（CLI > EVAL_BATCH_SIZE > BASE）。
# 早停与 BLEU 采样默认值：BASE_TRAINING_DEFAULTS.train_min_epochs / train_early_stop_patience / train_bleu4_max_samples。

_TASK_CONFIGS_BUILTIN: Dict[int, TaskConfig] = {
    1: {
        "auxiliary": "AM_Electronics",
        "target": "AM_CDs",
        "lr": 5e-4,
        "coef": 1,
    },
    2: {
        "auxiliary": "AM_Movies",
        "target": "AM_CDs",
        "lr": 1e-3,
        "coef": 0.1,
    },
    3: {
        "auxiliary": "AM_CDs",
        "target": "AM_Electronics",
        "lr": 5e-4,
        "coef": 0.5,
    },
    4: {
        "auxiliary": "AM_Movies",
        "target": "AM_Electronics",
        "lr": 1e-3,
        "coef": 0.5,
    },
    5: {
        "auxiliary": "AM_CDs",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
    },
    6: {
        "auxiliary": "AM_Electronics",
        "target": "AM_Movies",
        "lr": 1e-3,
        "coef": 0.5,
    },
    7: {
        "auxiliary": "Yelp",
        "target": "TripAdvisor",
        "lr": 1e-4,
        "coef": 0.5,
    },
    8: {
        "auxiliary": "TripAdvisor",
        "target": "Yelp",
        "lr": 5e-4,
        "coef": 1,
    },
}

task_configs: Dict[int, TaskConfig] = _TASK_CONFIGS_BUILTIN
TASK_DEFAULTS: Dict[int, TaskConfig] = task_configs


def resolve_task_idx_from_aux_target(auxiliary: str, target: str) -> Optional[int]:
    """由 auxiliary/target 反查任务号 1–8；未知组合返回 None。"""
    for tid, cfg in task_configs.items():
        if cfg["auxiliary"] == auxiliary and cfg["target"] == target:
            return int(tid)
    return None


# ---------------------------------------------------------------------------
# 命名训练预设（TRAINING_PRESETS）为空表；用户级训练配置来自 configs/odcr.yaml。
# 实际训练切片以 ``config_resolver.resolve_config`` → ``ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON`` 为准。
#
# 下列 get_train_batch_size / get_epochs 仅供**无 odcr CLI** 的极薄辅助脚本（如离线工具），
# 仅使用 BASE_TRAINING_DEFAULTS；用户级训练数值只允许走 configs/odcr.yaml 或 CLI --set。
# ---------------------------------------------------------------------------

_TRAINING_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "train_label_max_length",
        "epochs",
        "full_bleu_eval",
        "min_lr_ratio",
        "lr",
        "coef",
        "explainer_loss_weight",
        "emsize",
        "nlayers",
        "nhead",
        "nhid",
        "dropout",
        "per_device_train_batch_size",
        "per_gpu_batch_size",
        "train_dynamic_padding",
        "loss_weight_repeat_ul",
        "loss_weight_terminal_clean",
        "loss_weight_batch_diversity",
        "batch_diversity_warmup_epochs",
        "batch_diversity_mode",
        "batch_diversity_ema_init_mode",
        "batch_diversity_eps",
        "batch_diversity_use_ema",
        "batch_diversity_ema_decay",
        "batch_diversity_min_valid_tokens",
        "batch_diversity_loss_clamp_abs",
        "batch_diversity_ramp_epochs",
        "batch_diversity_ramp_target_scale",
        "uncertainty_entropy_eps",
        "uncertainty_high_entropy_threshold",
        "checkpoint_selection_mode",
        "checkpoint_guard_valid_loss_rel_tol",
        "checkpoint_guard_valid_loss_abs_tol",
        "checkpoint_composite_w_bleu4",
        "checkpoint_composite_w_rouge_l",
        "checkpoint_composite_w_meteor",
        "checkpoint_composite_w_dist1",
        "checkpoint_composite_w_dist2",
        "checkpoint_composite_w_dirty",
        "ema_decay",
        "terminal_clean_span",
        "full_bleu_decode_strategy",
        "ema_enabled",
        "generate_during_train",
        "checkpoint_metric",
        "decode_backend",
        "decode_backend_fallback_policy",
        "train_time_eval_decode_backend",
        "ddp_find_unused_parameters",
        "step5_strict_index_batches",
        # Step5：可与 decode 预设对齐的生成/解释支路标量（可选写入 training 行以按 task 覆盖）
        "decode_top_k",
        "gap_threshold",
        "generate_temperature",
        "label_smoothing",
        "max_explanation_length",
        "prefix_greedy_steps",
        "repetition_penalty",
        "train_mode",
        "train_precision",
        "per_device_eval_batch_size",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lambda_ortho",
        "lambda_ortho_xcov",
        "lambda_ortho_cos",
        "lambda_ortho_step5",
    }
)

_TRAINING_PRESET_INT_KEYS: FrozenSet[str] = frozenset(
    {
        "train_batch_size",
        "train_label_max_length",
        "epochs",
        "per_device_train_batch_size",
        "per_gpu_batch_size",
        "terminal_clean_span",
        "batch_diversity_warmup_epochs",
        "batch_diversity_ramp_epochs",
        "batch_diversity_min_valid_tokens",
        "decode_top_k",
        "max_explanation_length",
        "prefix_greedy_steps",
        "emsize",
        "nlayers",
        "nhead",
        "nhid",
        "lora_r",
        "per_device_eval_batch_size",
    }
)

_TRAINING_PRESET_FLOAT_KEYS: FrozenSet[str] = frozenset(
    {
        "min_lr_ratio",
        "lr",
        "coef",
        "explainer_loss_weight",
        "dropout",
        "loss_weight_repeat_ul",
        "loss_weight_terminal_clean",
        "loss_weight_batch_diversity",
        "batch_diversity_eps",
        "batch_diversity_ema_decay",
        "batch_diversity_loss_clamp_abs",
        "batch_diversity_ramp_target_scale",
        "uncertainty_entropy_eps",
        "uncertainty_high_entropy_threshold",
        "checkpoint_guard_valid_loss_rel_tol",
        "checkpoint_guard_valid_loss_abs_tol",
        "checkpoint_composite_w_bleu4",
        "checkpoint_composite_w_rouge_l",
        "checkpoint_composite_w_meteor",
        "checkpoint_composite_w_dist1",
        "checkpoint_composite_w_dist2",
        "checkpoint_composite_w_dirty",
        "ema_decay",
        "gap_threshold",
        "generate_temperature",
        "label_smoothing",
        "repetition_penalty",
        "lora_alpha",
        "lora_dropout",
        "lambda_ortho",
        "lambda_ortho_xcov",
        "lambda_ortho_cos",
        "lambda_ortho_step5",
    }
)

_TRAINING_PRESET_BOOL_KEYS: FrozenSet[str] = frozenset(
    {
        "train_dynamic_padding",
        "ema_enabled",
        "generate_during_train",
        "ddp_find_unused_parameters",
        "batch_diversity_use_ema",
        "step5_strict_index_batches",
    }
)

def _normalize_training_checkpoint_metric_yaml(raw: Any, *, ctx: str) -> str:
    """训练 preset 行内 checkpoint_metric：仅 valid_loss（别名 loss）。"""
    v = str(raw).strip().lower() if raw is not None else "valid_loss"
    if v == "loss":
        v = "valid_loss"
    if v != "valid_loss":
        raise ValueError(
            f"{ctx}: checkpoint_metric 仅允许 valid_loss（历史别名 loss），当前为 {raw!r}；"
            "禁止以 bleu4 等单一生成指标作为 checkpoint_metric；复合 tie-break 由 training_row.checkpoint_selection_mode=guarded_composite 控制。"
        )
    return "valid_loss"


def _validate_training_presets(presets: Dict[str, Any], *, name: str = "TRAINING_PRESETS") -> None:
    """模块加载时校验：任务键 1..8、字段名合法、数值类型基本合理。"""

    def _check_row(row: Dict[str, Any], ctx: str) -> None:
        unknown = set(row.keys()) - _TRAINING_PRESET_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{name} {ctx} 含有未知字段 {sorted(unknown)}；"
                f"允许: {sorted(_TRAINING_PRESET_ALLOWED_KEYS)}"
            )
        for k, v in row.items():
            if k == "full_bleu_eval":
                parse_full_bleu_eval_block(v, ctx=f"{name} {ctx} full_bleu_eval")
                continue
            if k == "full_bleu_decode_strategy":
                parse_full_bleu_decode_strategy(v, ctx=f"{name} {ctx} full_bleu_decode_strategy")
                continue
            if k == "checkpoint_metric":
                _normalize_training_checkpoint_metric_yaml(v, ctx=f"{name} {ctx} checkpoint_metric")
                continue
            if k == "checkpoint_selection_mode":
                parse_checkpoint_selection_mode(v, ctx=f"{name} {ctx} checkpoint_selection_mode")
                continue
            if k == "batch_diversity_mode":
                parse_batch_diversity_mode(v, ctx=f"{name} {ctx} batch_diversity_mode")
                continue
            if k == "batch_diversity_ema_init_mode":
                parse_batch_diversity_ema_init_mode(
                    v, ctx=f"{name} {ctx} batch_diversity_ema_init_mode"
                )
                continue
            if k == "train_mode":
                parse_train_mode(v, ctx=f"{name} {ctx} train_mode")
                continue
            if k == "train_precision":
                parse_train_precision(v, ctx=f"{name} {ctx} train_precision")
                continue
            if k == "lora_target_modules":
                if v is None:
                    continue
                if not isinstance(v, list):
                    raise TypeError(
                        f"{name} {ctx} 字段 lora_target_modules 须为 str 列表，当前为 {type(v).__name__}"
                    )
                for it in v:
                    if not isinstance(it, str) or not str(it).strip():
                        raise ValueError(
                            f"{name} {ctx} lora_target_modules 须为非空字符串列表，非法项: {it!r}"
                        )
                continue
            if k in _TRAINING_PRESET_INT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为整数类型，当前为 {type(v).__name__}")
                if isinstance(v, float) and not float(v).is_integer():
                    raise ValueError(f"{name} {ctx} 字段 {k!r} 应为整数值，当前为 {v!r}")
            elif k in _TRAINING_PRESET_FLOAT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为数值类型，当前为 {type(v).__name__}")
                fv = float(v)
                if not math.isfinite(fv):
                    raise ValueError(f"{name} {ctx} 字段 {k!r} 应为有限数值，当前为 {v!r}")
            elif k in _TRAINING_PRESET_BOOL_KEYS:
                if not isinstance(v, bool):
                    raise TypeError(f"{name} {ctx} 字段 {k!r} 应为 bool，当前为 {type(v).__name__}")

    for preset_name, blob in presets.items():
        if not isinstance(blob, dict):
            raise TypeError(f'{name} 中预设 {preset_name!r} 应为 dict，当前为 {type(blob).__name__}')
        keys = list(blob.keys())
        if keys and all(isinstance(k, int) and 1 <= k <= 8 for k in keys):
            for tid, row in blob.items():
                if not isinstance(row, dict):
                    raise TypeError(
                        f'{name} 预设 {preset_name!r} 的任务 {tid} 值应为 dict，当前为 {type(row).__name__}'
                    )
                _check_row(row, f"预设 {preset_name!r} task={tid}")
        else:
            _check_row(blob, f"预设 {preset_name!r}（全局）")


TRAINING_PRESETS: Dict[str, Any] = _load_training_presets_from_yaml_required()

_validate_training_presets(TRAINING_PRESETS)

TRAINING_ORCHESTRATION_FORBIDDEN_KEYS: FrozenSet[str] = frozenset(
    {
        "eval_device_policy",
        "poll_interval_sec",
        "compile_enabled",
        "compile_mode",
        "eval_profile",
        "decode_preset",
        "rerank_preset",
        "max_pending_jobs",
    }
)

_FORBIDDEN_TRAINING_KEY_HINTS: Dict[str, str] = {
    "eval_device_policy": "configs/odcr.yaml eval/daemon orchestration",
    "poll_interval_sec": "configs/odcr.yaml eval/daemon orchestration",
    "compile_enabled": "retired: no torch.compile consumer exists in the active ODCR path",
    "compile_mode": "retired: no torch.compile consumer exists in the active ODCR path",
    "max_pending_jobs": "configs/odcr.yaml eval/daemon orchestration",
    "eval_profile": "configs/odcr.yaml eval.profiles",
    "decode_preset": "configs/odcr.yaml eval.decode",
    "rerank_preset": "configs/odcr.yaml eval.rerank",
}


def format_forbidden_training_keys_error(bad: List[str], *, preset_chain: str) -> str:
    lines = [
        f"training config chain {preset_chain!r} contains orchestration-only keys {bad}; "
        "write those controls to configs/odcr.yaml ownership blocks, not training_row."
    ]
    for key in bad:
        lines.append(f"  - {key!r} belongs in: {_FORBIDDEN_TRAINING_KEY_HINTS.get(key, 'its owning One-Control block')}")
    return "\n".join(lines)


def assert_no_forbidden_training_keys(row: Mapping[str, Any], *, ctx: str) -> None:
    bad = sorted(key for key in TRAINING_ORCHESTRATION_FORBIDDEN_KEYS if key in row)
    if bad:
        raise ValueError(format_forbidden_training_keys_error(bad, preset_chain=ctx))

# ---------------------------------------------------------------------------
# 命名 hardware 预设（内置校验表；主链由 configs/odcr.yaml 解析）
#
# 主线：父进程 ``config_resolver.resolve_config`` 选定 stem，序列化硬件切片写入 ``ODCR_HARDWARE_PROFILE_JSON``，并注入
# ``ODCR_HARDWARE_PRESET``（stem 字符串）。torchrun 子进程 **只消费** 上述注入结果，不再把父 shell 的零散 ENV
# 当作二次选型入口。
#
# torchrun 子进程缺少 JSON 注入时直接 fail-fast，避免零散 ENV/CLI 变成隐形控制面。
# ---------------------------------------------------------------------------

_HARDWARE_PRESET_ALLOWED_KEYS: FrozenSet[str] = frozenset(
    {
        "max_parallel_cpu",
        "num_proc",
        "max_num_proc",
        "reserved_cpu",
        "tokenization_num_proc",
        "num_proc_configured",
        "tokenization_num_proc_source",
        "tokenization_num_proc_formula",
        "worker_budget_formula",
        "ddp_world_size",
        "dataloader_num_workers_train",
        "dataloader_num_workers_valid",
        "dataloader_num_workers_test",
        "dataloader_prefetch_factor_train",
        "dataloader_prefetch_factor_valid",
        "dataloader_prefetch_factor_test",
        "dataloader_workers_train_per_rank_cap",
        "omp_num_threads",
        "mkl_num_threads",
        "tokenizers_parallelism",
        "pin_memory",
        "persistent_workers",
        "non_blocking_h2d",
        # launcher-only：仅出现在 YAML/命名表校验；不入 ODCR_HARDWARE_PROFILE_JSON 归一化结果
        "cuda_visible_devices",
    }
)

_HARDWARE_PRESET_INT_KEYS: FrozenSet[str] = _HARDWARE_PRESET_ALLOWED_KEYS - frozenset(
    {
        "tokenizers_parallelism",
        "pin_memory",
        "persistent_workers",
        "non_blocking_h2d",
        "cuda_visible_devices",
        "num_proc_configured",
        "tokenization_num_proc_source",
        "tokenization_num_proc_formula",
        "worker_budget_formula",
    }
)

_HARDWARE_PRESET_REQUIRED_CHILD_KEYS: FrozenSet[str] = frozenset(
    {
        "ddp_world_size",
        "num_proc",
        "max_num_proc",
        "reserved_cpu",
        "max_parallel_cpu",
        "dataloader_num_workers_train",
        "dataloader_num_workers_valid",
        "dataloader_num_workers_test",
        "dataloader_prefetch_factor_train",
        "dataloader_prefetch_factor_valid",
        "dataloader_prefetch_factor_test",
        "pin_memory",
        "persistent_workers",
        "non_blocking_h2d",
    }
)


def _normalize_hardware_profile_mapping(obj: Mapping[str, Any]) -> Dict[str, Any]:
    """将 ODCR_HARDWARE_PROFILE_JSON 或同类 mapping 规范为 build_resolved 可用的 hardware 切片。"""
    out: Dict[str, Any] = {}
    if not isinstance(obj, Mapping):
        return out
    for k, v in obj.items():
        sk = str(k)
        if sk not in _HARDWARE_PRESET_ALLOWED_KEYS:
            continue
        if sk == "cuda_visible_devices":
            continue
        if sk in ("tokenizers_parallelism", "pin_memory", "persistent_workers", "non_blocking_h2d"):
            if isinstance(v, bool):
                out[sk] = v
            else:
                out[sk] = str(v).strip().lower() in ("true", "1", "yes", "on")
            continue
        try:
            iv = int(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        out[sk] = iv
    return out


def _validate_hardware_presets(presets: Dict[str, Any], *, name: str = "HARDWARE_PRESETS") -> None:
    for preset_name, blob in presets.items():
        if not isinstance(blob, dict):
            raise TypeError(f'{name} 中预设 {preset_name!r} 应为 dict，当前为 {type(blob).__name__}')
        unknown = set(blob.keys()) - _HARDWARE_PRESET_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{name} 预设 {preset_name!r} 含有未知字段 {sorted(unknown)}；"
                f"允许: {sorted(_HARDWARE_PRESET_ALLOWED_KEYS)}"
            )
        for k, v in blob.items():
            if k == "cuda_visible_devices":
                if not isinstance(v, str) or not str(v).strip():
                    raise TypeError(
                        f"{name} 预设 {preset_name!r} 字段 cuda_visible_devices 须为非空 str，当前为 {type(v).__name__}"
                    )
                continue
            if k in ("tokenizers_parallelism", "pin_memory", "persistent_workers", "non_blocking_h2d"):
                if not isinstance(v, bool):
                    raise TypeError(
                        f"{name} 预设 {preset_name!r} 字段 {k} 须为 bool，当前为 {type(v).__name__}"
                    )
                continue
            if k in _HARDWARE_PRESET_INT_KEYS:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise TypeError(f"{name} 预设 {preset_name!r} 字段 {k!r} 应为整数类型，当前为 {type(v).__name__}")
                if isinstance(v, float) and not float(v).is_integer():
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 应为整数值，当前为 {v!r}")
                iv = int(v)
                if iv < 0 and k != "num_proc":  # num_proc 必须 >=1，单独在值上约束
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 不可为负，当前为 {iv}")
                if k == "num_proc" and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} num_proc 须 >= 1，当前为 {iv}")
                if k == "ddp_world_size" and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} ddp_world_size 须 >= 1，当前为 {iv}")
                if k in ("omp_num_threads", "mkl_num_threads") and iv < 1:
                    raise ValueError(f"{name} 预设 {preset_name!r} 字段 {k!r} 须 >= 1，当前为 {iv}")


_HARDWARE_PRESETS_BUILTIN: Dict[str, Dict[str, Any]] = {
    "gpu01_single_12c": {
        "max_parallel_cpu": 10,
        "num_proc": 6,
        "dataloader_num_workers_train": 4,
        "dataloader_num_workers_valid": 2,
        "dataloader_num_workers_test": 2,
        "dataloader_prefetch_factor_train": 2,
        "dataloader_prefetch_factor_valid": 2,
        "dataloader_prefetch_factor_test": 2,
    },
    "gpu01_ddp2_12c": {
        "max_parallel_cpu": 10,
        "num_proc": 4,
        "dataloader_workers_train_per_rank_cap": 3,
        "dataloader_num_workers_valid": 2,
        "dataloader_num_workers_test": 2,
        "dataloader_prefetch_factor_train": 2,
        "dataloader_prefetch_factor_valid": 2,
        "dataloader_prefetch_factor_test": 2,
    },
}

HARDWARE_PRESETS: Dict[str, Dict[str, Any]] = _HARDWARE_PRESETS_BUILTIN

_validate_hardware_presets(HARDWARE_PRESETS)

def _active_hardware_preset_slice() -> Optional[Dict[str, Any]]:
    """当前激活的 hardware 切片：仅接受 resolver 注入的 ODCR_HARDWARE_PROFILE_JSON。"""
    rawj = (os.environ.get("ODCR_HARDWARE_PROFILE_JSON") or "").strip()
    if not rawj:
        raise RuntimeError(
            "缺少 ODCR_HARDWARE_PROFILE_JSON：hardware 值必须由 config_resolver 从 configs/odcr.yaml 解析后注入；"
            "MAX_PARALLEL_CPU、ODCR_NUM_PROC 和 child --num-proc 不再作为 active fallback。"
        )
    try:
        loaded = json.loads(rawj)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ODCR_HARDWARE_PROFILE_JSON 非法 JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("ODCR_HARDWARE_PROFILE_JSON 根须为 object。")
    norm = _normalize_hardware_profile_mapping(loaded)
    if not norm:
        raise RuntimeError("ODCR_HARDWARE_PROFILE_JSON 无法解析为有效 hardware 切片。")
    missing = sorted(key for key in _HARDWARE_PRESET_REQUIRED_CHILD_KEYS if key not in norm)
    if missing:
        raise RuntimeError(
            "ODCR_HARDWARE_PROFILE_JSON 缺少必需 hardware 字段: "
            + ", ".join(missing)
            + "；请通过 config_resolver 从 configs/odcr.yaml 注入完整 payload。"
        )
    return norm


def training_preset_is_per_task() -> bool:
    """历史 API：环境变量不再驱动训练预设切片；恒为 False。"""
    return False


def get_task_config(task_idx: int) -> Optional[TaskConfig]:
    """仅返回 TASK_DEFAULTS 表项（不含预设合并）；训练时以 build_resolved_training_config 为准。"""
    return TASK_DEFAULTS.get(int(task_idx))


NO_ACCUM_BATCH_SEMANTICS_VERSION = "odcr_no_accum/1"
NO_ACCUM_REMOVED_MESSAGE = (
    "grad_accum has been removed in ODCR no-accum architecture; use per_gpu_batch_size "
    "and global_batch_size = per_gpu_batch_size * ddp_world_size."
)
_RETIRED_ACCUMULATION_ENV_NAMES = (
    "ODCR_GRAD_ACCUM",
    "ODCR_GRADIENT_ACCUMULATION_STEPS",
    "ODCR_ACCUMULATE_GRAD_BATCHES",
    "ODCR_ACCUM_STEPS",
    "ODCR_ACCUMULATION_STEPS",
)
_RETIRED_ACCUMULATION_ROW_KEYS = frozenset(
    {
        "grad_accum",
        "gradient_accumulation_steps",
        "accumulate_grad_batches",
        "accum_steps",
        "accumulation_steps",
    }
)


def _reject_removed_accumulation_env() -> None:
    present = [name for name in _RETIRED_ACCUMULATION_ENV_NAMES if (os.environ.get(name) or "").strip()]
    if present:
        raise RuntimeError(f"Retired accumulation environment variable(s) set: {present}. {NO_ACCUM_REMOVED_MESSAGE}")


def _reject_removed_accumulation_row(row: Mapping[str, Any]) -> None:
    present = sorted(str(key) for key in row if str(key) in _RETIRED_ACCUMULATION_ROW_KEYS)
    if present:
        raise RuntimeError(f"Resolved training payload contains retired accumulation key(s) {present}. {NO_ACCUM_REMOVED_MESSAGE}")


def resolve_train_batch_layout(
    global_batch_size: int,
    world_size: int,
    *,
    per_device_batch_size: Optional[int] = None,
) -> Tuple[int, int, int]:
    """Resolve no-accum train batch layout: G = per_gpu_batch_size × world_size."""
    G = int(global_batch_size)
    W = int(world_size)
    if G < 1:
        raise ValueError(f"global_batch_size 须 >= 1，当前为 {G}")
    if W < 1:
        raise ValueError(f"world_size 须 >= 1，当前为 {W}")
    if per_device_batch_size is None:
        raise ValueError("no-accum batch layout requires per_gpu_batch_size/per_device_train_batch_size")
    P = max(1, int(per_device_batch_size))
    expected = P * W
    if G != expected:
        raise ValueError(
            f"global_batch_size={G} must equal per_gpu_batch_size({P}) * ddp_world_size({W}) = {expected}. "
            f"ODCR uses {NO_ACCUM_BATCH_SEMANTICS_VERSION}."
        )
    return G, P, expected


def resolve_step3_no_accum_batch_layout(row: Mapping[str, Any], world_size: int) -> Tuple[int, int, int]:
    _reject_removed_accumulation_row(row)
    G = _preset_int_min(row.get("train_batch_size", BASE_TRAINING_DEFAULTS.train_batch_size), 1)
    if "per_device_train_batch_size" not in row:
        raise ValueError("no-accum batch layout requires per_device_train_batch_size in resolved payload.")
    P = max(1, int(row["per_device_train_batch_size"]))
    return resolve_train_batch_layout(G, world_size, per_device_batch_size=P)


# 历史名：与 resolve_train_batch_layout 同义（单一实现，避免两套公式漂移）
resolve_ddp_train_microbatch_layout = resolve_train_batch_layout


def resolve_eval_batch_layout(eval_batch_size: int, ddp_world_size: int) -> Tuple[int, int]:
    """
    全局评测 batch 按 DDP world size 切分；返回 (global_eval_batch_size, eval_per_gpu_batch_size)。
    与 step5_engine / step3_train_core 中 eval strict 校验一致；step4 反事实推理使用同一合同。
    """
    E = int(eval_batch_size)
    W = int(ddp_world_size)
    if E < 1:
        raise ValueError(f"eval_batch_size 须 >= 1，当前为 {E}")
    if W < 1:
        raise ValueError(f"ddp_world_size 须 >= 1，当前为 {W}")
    if E % W != 0:
        raise ValueError(
            f"eval_batch_size={E} 与 world_size={W} 不整除。"
            "请修改 configs/odcr.yaml 中 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
        )
    return E, E // W


def resolve_train_batch_from_training_row(
    row: Mapping[str, Any],
    world_size: int,
) -> Tuple[int, int, int]:
    """
    与 ``build_resolved_training_config`` 中 batch 解析一致；返回
    ``(global_batch_size, per_gpu_batch_size, effective_global_batch_size)``。
    """
    _reject_removed_accumulation_row(row)
    G = int(BASE_TRAINING_DEFAULTS.train_batch_size)
    if "train_batch_size" in row:
        G = _preset_int_min(row["train_batch_size"], 1)
    p_opt: Optional[int] = None
    if "per_gpu_batch_size" in row:
        p_opt = max(1, int(row["per_gpu_batch_size"]))
    elif "per_device_train_batch_size" in row:
        p_opt = max(1, int(row["per_device_train_batch_size"]))
    G, P, eff = resolve_train_batch_layout(
        G,
        world_size,
        per_device_batch_size=p_opt,
    )
    return G, P, eff


def get_embed_batch_size():
    """返回 embedding 计算的 batch_size，供 preprocess_b / compute_embeddings 使用"""
    return embed_batch_size


def get_eval_batch_size(cli: Optional[int] = None) -> int:
    """辅脚本用 eval batch；优先级：cli > EVAL_BATCH_SIZE > BASE。odcr torchrun 子进程请用 effective payload。"""
    if cli is not None:
        return max(1, int(cli))
    return max(
        1,
        int(
            os.environ.get(
                "EVAL_BATCH_SIZE",
                str(BASE_TRAINING_DEFAULTS.eval_batch_size),
            )
        ),
    )


def _first_env_int_max1(names: Tuple[str, ...]) -> Optional[int]:
    for n in names:
        if n in os.environ:
            return max(1, int(os.environ[n]))
    return None


def _fixed_dataloader_prefetch_factor(num_workers: int) -> Optional[int]:
    if num_workers <= 0:
        return None
    return max(2, 4)


def _auto_derive_dataloader_num_workers(split: str, cap_parallel: int) -> int:
    """无 hardware preset / 无对应 ENV 时的 DataLoader workers 推导（与历史逻辑一致）。"""
    _cap = int(cap_parallel)
    n = _get_num_cpu()
    cap_t = min(_cap, 16)
    cap_v = min(max(4, _cap // 2), 8)
    if split == "train":
        return min(max(2, n // 2), cap_t)
    if split in ("valid", "test"):
        return min(max(1, n // 4), cap_v)
    return min(max(1, n // 4), cap_v)


def _resolve_dataloader_num_workers_for_split(split: str, cli: Optional[int]) -> int:
    """resolver-injected hardware profile plus deterministic derivation; no env/CLI fallback."""
    _ = cli
    mp = _resolve_max_parallel_cpu_cli(None)
    nw = _auto_derive_dataloader_num_workers(split, mp)
    rp = _active_hardware_preset_slice()
    wkey = f"dataloader_num_workers_{split}" if split in ("train", "valid", "test") else None
    if rp and wkey and wkey in rp:
        nw = max(0, int(rp[wkey]))
    return nw


def _resolve_num_proc_cli(num_proc_cli: Optional[int]) -> int:
    """resolver-injected hardware profile only; child --num-proc is transport-equality checked by entries."""
    _ = num_proc_cli
    mp = _resolve_max_parallel_cpu_cli(None)
    v = min(_get_num_cpu(), mp)
    rp = _active_hardware_preset_slice()
    if rp and "num_proc" in rp:
        v = max(1, int(rp["num_proc"]))
        v = min(v, _get_num_cpu())
        return v
    raise RuntimeError("ODCR_HARDWARE_PROFILE_JSON 缺少 num_proc；hardware 值必须由 resolver 注入。")


def _resolve_ddp_train_num_workers_per_rank_cli(world_size: int, cli_cap: Optional[int]) -> int:
    _ = cli_cap
    ws = max(int(world_size), 1)
    dl_train = _resolve_dataloader_num_workers_for_split("train", None)
    share = max(1, _resolve_max_parallel_cpu_cli(None) // ws)
    out = max(1, min(dl_train, share))
    rp = _active_hardware_preset_slice()
    if rp and "dataloader_workers_train_per_rank_cap" in rp:
        cap = max(1, int(rp["dataloader_workers_train_per_rank_cap"]))
        out = max(1, min(out, cap))
    return out


def get_num_proc() -> int:
    """datasets.map（Tokenize）并行进程数；与 DataLoader num_workers 独立。"""
    return _resolve_num_proc_cli(None)


def get_max_parallel_cpu() -> int:
    """并行 CPU 上限；主线来自 resolver-injected hardware profile。"""
    return _get_max_parallel_cpu()


def get_dataloader_num_workers(split="train"):
    """
    PyTorch DataLoader 的 num_workers，与 datasets.map 的 num_proc 独立。
    split: 'train' | 'valid' | 'test'。
    """
    return _resolve_dataloader_num_workers_for_split(str(split), None)


@contextmanager
def hf_datasets_progress_bar(enabled: bool):
    """仅 rank0 显示 datasets.map 的 tqdm；其他 rank 关闭，避免 torchrun 多进程重复进度条与日志。"""
    if enabled:
        yield
        return
    try:
        from datasets.utils.logging import disable_progress_bar
    except ImportError:
        yield
        return
    disable_progress_bar()
    yield


def get_dataloader_prefetch_factor(num_workers: int, split: Optional[str] = None):
    """
    num_workers==0 时为 None。
    若给定 split 且 resolver-injected hardware profile 有值则用之；否则为 prefetch=4。
    """
    if num_workers <= 0:
        return None
    rp = _active_hardware_preset_slice()
    if split == "train" and rp and "dataloader_prefetch_factor_train" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_train"]))
    if split in ("valid", "eval") and rp and "dataloader_prefetch_factor_valid" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_valid"]))
    if split == "test" and rp and "dataloader_prefetch_factor_test" in rp:
        return max(2, int(rp["dataloader_prefetch_factor_test"]))
    return _fixed_dataloader_prefetch_factor(num_workers)


def get_ddp_train_num_workers_per_rank(world_size: int) -> int:
    """
    DDP 下每个训练进程的 DataLoader worker 数。
    world_size × workers 不超过 max_parallel_cpu 均分份额；可用 dataloader_workers_train_per_rank_cap 收紧。
    """
    return _resolve_ddp_train_num_workers_per_rank_cli(world_size, None)


@dataclass(frozen=True)
class FinalTrainingConfig:
    """入口 resolve 之后的冻结训练配置；训练循环只读此对象。"""

    task_idx: int
    auxiliary: str
    target: str
    scenario: str
    direction: str
    task_profile_id: str
    task_profile_key: str
    profile_isolation_hash: str
    preset_name: Optional[str]
    world_size: int
    sources: Tuple[Tuple[str, str], ...]

    learning_rate: float
    scheduler_initial_lr: float
    initial_lr: float
    epochs: int
    max_epochs: int
    validate_every_epochs: int
    max_grad_norm: float
    tokenizer_max_length: int
    evidence_max_length: int
    valid_batch_size: int
    valid_micro_batch_size: int

    train_batch_size: int
    global_batch_size: int
    batch_size_global: int
    batch_size: int
    per_device_train_batch_size: int
    per_gpu_batch_size: int
    effective_global_batch_size: int
    batch_semantics_version: str
    grad_accum_removed: bool

    num_proc: int
    max_parallel_cpu: int
    hardware_preset_name: Optional[str]
    dataloader_num_workers_train: int
    dataloader_num_workers_valid: int
    dataloader_num_workers_test: int
    dataloader_prefetch_factor_train: Optional[int]
    dataloader_prefetch_factor_valid: Optional[int]
    dataloader_prefetch_factor_test: Optional[int]
    pin_memory: bool
    persistent_workers: bool
    non_blocking_h2d: bool

    min_lr_ratio: float
    lr_scheduler: str
    scheduler_type: str
    warmup_epochs: float
    odcr_warmup_steps: Optional[int]
    odcr_warmup_ratio: Optional[float]
    optimizer_config_json: str
    precision_config_json: str
    tokenizer_config_json: str
    evidence_config_json: str
    scheduler_config_json: str
    valid_batch_config_json: str
    scenario_profile_json: str
    task_profile_config_json: str
    backup_profiles_config_json: str
    exploration_profiles_config_json: str
    worker_profiles_config_json: str
    prefetcher_config_json: str
    checkpoint_policy_config_json: str
    quality_gate_config_json: str
    grad_finite_config_json: str
    numerical_stability_config_json: str
    diagnostic_eval_config_json: str
    cross_rank_structured_gather_config_json: str
    memory_config_json: str
    timing_config_json: str
    performance_candidates_config_json: str
    cache_policy_config_json: str
    objective_drift_config_json: str
    recovery_config_json: str
    phase_loss_schedule_config_json: str
    conflict_aware_config_json: str
    loss_gradient_conflict_probe_config_json: str
    adapter_gating_config_json: str
    paper_candidate_selection_config_json: str
    checkpoint_averaging_config_json: str

    eval_batch_size: int
    min_epochs: int
    train_min_epochs: int
    early_stop_patience: int
    early_stop_patience_full: int
    early_stop_patience_loss: int

    full_bleu_eval_resolved: FullBleuEvalResolved

    checkpoint_metric: str
    dual_bleu_eval: bool
    bleu4_max_samples: int
    quick_eval_max_samples: int

    coef: float
    explainer_loss_weight: float
    # 训练期 full BLEU 监控解码：greedy / inherit（inherit=与 decode_strategy 一致）；非正式 eval 口径
    full_bleu_decode_strategy: str = "inherit"
    ema_enabled: bool = True
    ema_decay: float = 0.999
    generate_during_train: bool = False
    decode_backend: str = "sdpa_kv_fast"
    decode_backend_fallback_policy: str = "raise"
    train_time_eval_decode_backend: str = "sdpa_kv_safe"

    emsize: int = 1024
    nlayers: int = 2
    nhid: int = 2048
    # 词表大小在 odcr 主入口用 len(tokenizer) 覆盖；此处仅作占位默认值
    ntoken: int = 32128
    dropout: float = 0.2
    nhead: int = 2
    label_smoothing: float = 0.1
    repetition_penalty: float = 1.15
    generate_temperature: float = 0.8
    generate_top_p: float = 0.9
    max_explanation_length: int = 25
    train_label_max_length: int = 64
    train_dynamic_padding: bool = True
    train_padding_strategy: str = "dynamic_batch"
    decode_strategy: str = "greedy"
    decode_seed: Optional[int] = None
    no_repeat_ngram_size: Optional[int] = None
    min_len: Optional[int] = None
    domain_fusion_mode: str = "gate_cross_attn"
    # --- decode controller v2（手写 generate；与 One-Control eval.decode 对齐）---
    soft_max_len: Optional[int] = None
    hard_max_len: Optional[int] = None
    eos_boost_start: int = 9999
    eos_boost_value: float = 0.0
    tail_temperature: float = -1.0
    tail_top_p: float = -1.0
    forbid_eos_after_open_quote: bool = True
    forbid_eos_after_open_bracket: bool = True
    forbid_bad_terminal_tokens: bool = True
    bad_terminal_token_ids: Optional[Tuple[int, ...]] = None
    decode_token_repeat_window: int = 4
    decode_token_repeat_max: int = 2
    candidate_family: str = "balanced"
    candidate_mixed_include_diverse: bool = True
    # --- explanation 支路轻量质量正则 ---
    loss_weight_repeat_ul: float = 0.0
    loss_weight_terminal_clean: float = 0.0
    terminal_clean_span: int = 3
    loss_weight_batch_diversity: float = 0.02
    batch_diversity_warmup_epochs: int = 2
    batch_diversity_mode: str = "mean_prob_neg_entropy"
    batch_diversity_eps: float = 1e-8
    batch_diversity_use_ema: bool = True
    batch_diversity_ema_decay: float = 0.9
    batch_diversity_min_valid_tokens: int = 64
    batch_diversity_loss_clamp_abs: float = 0.2
    batch_diversity_ramp_epochs: int = 2
    batch_diversity_ramp_target_scale: float = 1.0
    batch_diversity_ema_init_mode: str = "uniform"
    # shared/specific 正交损失（Step3 主训 + Step5 keep）；xcov/cos 为 build_orthogonal_losses 内组合系数
    lambda_ortho: float = 0.2
    lambda_ortho_xcov: float = 1.0
    lambda_ortho_cos: float = 0.25
    lambda_ortho_step5: float = 0.15
    # Step5A/Step5B：LCI 与 FCA 权重来自 step5.lci / step5.fca。
    step5_lci_weight: float = 0.12
    step5_fca_weight: float = 0.08
    method_name: str = "CSB-ODCR"
    method_full_name: str = "CSB-ODCR: Causal Structure Bottleneck for Orthogonal Disentangled Counterfactual Recommendation"
    method_family: str = "csb_odcr"
    experiment_profile: str = "csb_odcr_sidecar_stable"
    ablation_profile: str = "csb_odcr_sidecar_stable"
    method_config_json: str = ""
    experiment_profile_config_json: str = ""
    experiment_profiles_config_json: str = ""
    csb_odcr_config_json: str = ""
    csb_contract_config_json: str = ""
    csb_conflict_routing_config_json: str = ""
    controlled_injection_config_json: str = ""
    step5_innovation_config_json: str = ""
    step3_structured_loss_weights_json: str = ""
    step3_loss_semantics_json: str = ""
    step3_upstream_evidence_json: str = ""
    step3_tokenizer_cache_manifest_json: str = ""
    step4_export_lineage_json: str = ""
    uncertainty_entropy_eps: float = 1e-8
    uncertainty_high_entropy_threshold: float = 1.0
    checkpoint_selection_mode: str = "guarded_composite"
    checkpoint_guard_valid_loss_rel_tol: float = 0.005
    checkpoint_guard_valid_loss_abs_tol: float = 0.0
    checkpoint_composite_w_bleu4: float = 0.32
    checkpoint_composite_w_rouge_l: float = 0.28
    checkpoint_composite_w_meteor: float = 0.22
    checkpoint_composite_w_dist1: float = 0.05
    checkpoint_composite_w_dist2: float = 0.05
    checkpoint_composite_w_dirty: float = 0.12
    gap_threshold: float = 0.35
    prefix_greedy_steps: int = 4
    decode_top_k: int = 5
    nuser: int = 0
    nitem: int = 0

    # Step5B：训练范式（manifest / config_resolved；train_mode=lora 时由 step5_native_lora 注入）
    train_mode: str = "lora"
    train_precision: str = "bf16"
    allow_tf32: bool = True
    amp_autocast: bool = True
    grad_scaler: bool = False
    per_device_eval_batch_size: int = 2
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ()

    device: int = 0
    device_ids: Tuple[int, ...] = ()
    save_file: str = ""
    # Step5：save_file = model/best.pth（canonical）。
    last_checkpoint_path: str = ""
    log_file: Optional[str] = None
    ddp_world_size: int = 1
    # True：DDP 安全默认（扫描未参与本步 loss 的参数，避免多分支图报错）。
    # False：吞吐向（降低 DDP 固定开销）；仅建议在计算图各步稳定、常开对抗等场景用预设显式关闭。
    ddp_find_unused_parameters: bool = True
    ddp_find_unused_false_preflight: str = "synthetic_one_batch"
    ddp_static_graph: bool = False
    ddp_graph_safety_preflight: bool = True
    # True：Step5 在首包上对 train/valid 再做一次 collate 后索引 min/max 审计（CPU fail-fast）
    step5_strict_index_batches: bool = False
    rank0_only_logging: bool = True
    run_id: str = ""
    ddp_fast_backends: bool = False

    logger: Any = None
    valid_dataset: Any = None

    def to_log_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("logger", None)
        d.pop("valid_dataset", None)
        if self.preset_name == "step3":
            d.pop("coef", None)
        _src = dict(self.sources)
        if self.preset_name == "step3":
            _src.pop("coef", None)
        d["sources"] = _src
        return d


def build_mainline_alignment_monitor_override(cfg: FinalTrainingConfig) -> Dict[str, Any]:
    """
    训练期 valid 全量文本监控 / guarded checkpoint：与当前 FinalTrainingConfig 主解码口径一致
    （含 uncertainty_low_temp_top_k 与 greedy 消融）；仅关闭尾部 temperature/top_p schedule。
    full_bleu_decode_strategy=greedy 时强制 greedy，便于与主 decode 对照消融。
    """
    from executors.decode_controller import resolve_decode_backend_name

    hm = int(getattr(cfg, "hard_max_len", None) or cfg.max_explanation_length)
    sm = getattr(cfg, "soft_max_len", None)
    soft = int(sm) if sm is not None and int(sm) > 0 else max(1, hm - 8)
    _nr = cfg.no_repeat_ngram_size
    nrs = max(3, int(_nr)) if _nr is not None and int(_nr) > 0 else 3
    _mn = cfg.min_len
    min_l = max(4, int(_mn)) if _mn is not None and int(_mn) > 0 else 4
    fb = str(getattr(cfg, "full_bleu_decode_strategy", "inherit")).strip().lower()
    if fb == "greedy":
        strat = "greedy"
        temp = 1.0
        top_p = 1.0
        gap_th = 0.35
        prefix_g = 4
        tk = 5
    else:
        strat = str(cfg.decode_strategy).strip().lower()
        temp = float(cfg.generate_temperature)
        top_p = float(cfg.generate_top_p)
        gap_th = float(getattr(cfg, "gap_threshold", 0.35))
        prefix_g = int(getattr(cfg, "prefix_greedy_steps", 4))
        tk = int(getattr(cfg, "decode_top_k", 5))
    return {
        "strategy": strat,
        "temperature": temp,
        "top_p": top_p,
        "gap_threshold": gap_th,
        "prefix_greedy_steps": prefix_g,
        "top_k": tk,
        "repetition_penalty": float(cfg.repetition_penalty),
        "no_repeat_ngram_size": nrs,
        "min_len": min_l,
        "soft_max_len": soft,
        "hard_max_len": max(1, hm),
        "eos_boost_start": int(getattr(cfg, "eos_boost_start", 9999)),
        "eos_boost_value": float(getattr(cfg, "eos_boost_value", 0.0)),
        "tail_temperature": -1.0,
        "tail_top_p": -1.0,
        "forbid_eos_after_open_quote": bool(getattr(cfg, "forbid_eos_after_open_quote", True)),
        "forbid_eos_after_open_bracket": bool(getattr(cfg, "forbid_eos_after_open_bracket", True)),
        "forbid_bad_terminal_tokens": bool(getattr(cfg, "forbid_bad_terminal_tokens", True)),
        "token_repeat_window": int(getattr(cfg, "decode_token_repeat_window", 4)),
        "token_repeat_max": int(getattr(cfg, "decode_token_repeat_max", 2)),
        "decode_seed": cfg.decode_seed,
        "uncertainty_entropy_eps": float(getattr(cfg, "uncertainty_entropy_eps", 1e-8)),
        "decode_backend": resolve_decode_backend_name(getattr(cfg, "train_time_eval_decode_backend", "sdpa_kv_safe")),
        "decode_run_context": "train_time_eval",
    }


def build_full_bleu_monitor_cfg_override(cfg: FinalTrainingConfig) -> Dict[str, Any]:
    """训练期 valid 文本监控与 guarded checkpoint 的 generate 覆盖：与主 decode 口径一致（见 build_mainline_alignment_monitor_override）。"""
    return build_mainline_alignment_monitor_override(cfg)


def format_full_bleu_monitor_log_line(cfg: FinalTrainingConfig) -> str:
    ov = build_mainline_alignment_monitor_override(cfg)
    return (
        f"[mainline_monitor] strategy={ov.get('strategy')} tail_noise=off "
        f"no_repeat_ngram>={ov.get('no_repeat_ngram_size')} min_len>={ov.get('min_len')} "
        f"(build_mainline_alignment_monitor_override)"
    )


def build_resolved_training_config(
    args: Any,
    *,
    task_idx: int,
    world_size: int,
    hardware_overrides: Optional[Dict[str, Any]] = None,
) -> FinalTrainingConfig:
    """
    **torchrun 子进程**构造 ``FinalTrainingConfig`` 的唯一入口。

    - 训练语义：必须存在 ``ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON``（父进程 ``config_resolver.resolve_config`` 写入的
      schema_version>=2 payload）；**禁止**再用 ``ODCR_TRAIN_PRESET`` / ``TRAINING_PRESETS`` / ``TRAIN_*`` 重算训练切片。
    - 硬件语义：须由父进程注入 ``ODCR_HARDWARE_PROFILE_JSON``（与 ``--hardware-preset`` / eval_profile 选定结果一致）；
      已注入时 **仅信任 JSON**，不再用 ``MAX_PARALLEL_CPU`` / ``ODCR_NUM_PROC`` 等父 shell 残留覆盖。
    - ``hardware_overrides``：保留为内部调用签名兼容；主线不读取它来覆盖 resolved hardware JSON。
    """
    _ = hardware_overrides
    _reject_removed_accumulation_env()
    src: Dict[str, str] = {}

    tid = int(task_idx)
    tc = TASK_DEFAULTS.get(tid)
    if tc is None:
        raise ValueError(f"无效 task_idx={tid}，TASK_DEFAULTS 中无此任务")

    _eff_raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
    if not _eff_raw:
        raise RuntimeError(
            "缺少 ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON：本进程须由 `python code/odcr.py …` 经 torchrun 启动。\n"
            "禁止裸调 executors/step5_entry、step3_entry 而未注入父进程 effective training payload。"
        )
    try:
        _payload = json.loads(_eff_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON 非合法 JSON: {e}") from e
    if int(_payload.get("task_id", -1)) != tid:
        raise ValueError(
            f"effective payload task_id={_payload.get('task_id')!r} 与当前解析 task_idx={tid} 不一致"
        )
    preset_nm = str(_payload.get("preset_name") or "").strip() or None
    row = _payload.get("training_row")
    if not isinstance(row, dict):
        raise TypeError("effective payload 缺少 training_row dict")

    def _reject_conflicting_child_cli(attr: str, expected: Any, payload_key: str, *, integer: bool = False) -> None:
        raw_cli = getattr(args, attr, None)
        if raw_cli is None:
            return
        if integer:
            ok = int(raw_cli) == int(expected)
        else:
            ok = math.isclose(float(raw_cli), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        if not ok:
            raise RuntimeError(
                "internal child argparse conflict: "
                f"--{attr.replace('_', '-')}={raw_cli!r} conflicts with "
                f"ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON.{payload_key}={expected!r}. "
                "Only public ./odcr --set may override configs/odcr.yaml; torchrun children must use the resolved payload."
            )

    _reject_conflicting_child_cli("epochs", row.get("epochs", BASE_TRAINING_DEFAULTS.epochs), "training_row.epochs", integer=True)
    _reject_conflicting_child_cli("learning_rate", row.get("lr"), "training_row.lr")
    if preset_nm == "step3":
        pass
    else:
        _reject_conflicting_child_cli("coef", row.get("coef"), "training_row.coef")
    assert_no_forbidden_training_keys(row, ctx="ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON.training_row")

    def _require_step3_row_key(key: str) -> Any:
        if preset_nm == "step3" and key not in row:
            raise RuntimeError(
                f"effective Step3 payload missing training_row.{key}; "
                "Step3 v0 controls must come from configs/odcr.yaml via config_resolver."
            )
        return row.get(key)

    auxiliary = str(_payload.get("auxiliary") or "").strip()
    target = str(_payload.get("target") or "").strip()
    if not auxiliary or not target:
        raise RuntimeError(
            "ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON 缺少 auxiliary/target；"
            "须由新版 python code/odcr.py 注入（schema_version>=2）。"
        )

    _hardware_json_injected = bool((os.environ.get("ODCR_HARDWARE_PROFILE_JSON") or "").strip())
    if not _hardware_json_injected:
        raise RuntimeError(
            "缺少 ODCR_HARDWARE_PROFILE_JSON：torchrun 子进程硬件语义必须由 "
            "config_resolver 从 configs/odcr.yaml 解析后注入。"
        )

    # ----- no-accum global/per-GPU train batch（仅父进程下发的 training_row；与 resolve_train_batch_from_training_row 共用）-----
    src["train_batch_size"] = "base"
    if "train_batch_size" in row:
        src["train_batch_size"] = "effective_payload"
    src["per_device_train_batch_size"] = "base"
    if "per_device_train_batch_size" in row:
        src["per_device_train_batch_size"] = "effective_payload"
    src["per_gpu_batch_size"] = "effective_payload" if "per_gpu_batch_size" in row else src["per_device_train_batch_size"]
    src["batch_semantics_version"] = "effective_payload"
    src["grad_accum_removed"] = "resolver-fixed"

    G, P, eff = resolve_train_batch_from_training_row(row, world_size)

    # ----- epochs -----
    ep = int(BASE_TRAINING_DEFAULTS.epochs)
    src["epochs"] = "base"
    if preset_nm == "step3":
        _require_step3_row_key("max_epochs")
    if "epochs" in row:
        ep = _preset_int_min(row["epochs"], 1)
        src["epochs"] = "effective_payload"
    if getattr(args, "epochs", None) is not None:
        src["epochs_cli_transport"] = "validated_equal_to_effective_payload"

    # ----- learning rate / coef：仅 training_row + 可选 CLI（禁止 TASK_DEFAULTS 二次合并）-----
    required_train_keys = ("lr",) if preset_nm == "step3" else ("lr", "coef")
    for _req in required_train_keys:
        if _req not in row:
            raise RuntimeError(
                f"ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON.training_row 缺少必需键 {_req!r}；"
                "须由 config_resolver.resolve_config 写入与 ResolvedConfig 一致的合并结果。"
            )
    initial_f = float(_coerce_task_param_numeric(row["lr"]))
    src["learning_rate"] = "effective_payload"

    # ----- min_lr_ratio -----
    min_lr = float(BASE_TRAINING_DEFAULTS.min_lr_ratio)
    src["min_lr_ratio"] = "base"
    _require_step3_row_key("min_lr_ratio")
    if "min_lr_ratio" in row:
        min_lr = float(row["min_lr_ratio"])
        src["min_lr_ratio"] = "effective_payload"

    # ----- lr_scheduler -----
    lr_sched = str(BASE_TRAINING_DEFAULTS.lr_scheduler)
    src["lr_scheduler"] = "base"
    _require_step3_row_key("lr_scheduler")
    if "lr_scheduler" in row and str(row["lr_scheduler"]).strip():
        v = str(row["lr_scheduler"]).strip().lower()
        if v in ("none", "off", "disabled"):
            lr_sched = "none"
        elif v in ("warmup_cosine", "warmup-cosine", "cosine"):
            lr_sched = "warmup_cosine"
        elif v in ("warmup_cosine_with_damping", "warmup-cosine-with-damping", "cosine_with_damping"):
            lr_sched = "warmup_cosine_with_damping"
        else:
            lr_sched = "none"
        src["lr_scheduler"] = "effective_payload"

    # ----- warmup_epochs -----
    wu_ep = float(BASE_TRAINING_DEFAULTS.warmup_epochs)
    src["warmup_epochs"] = "base"
    if preset_nm == "step3":
        wu_ep = 0.0
        src["warmup_epochs"] = "retired_inactive_step3"
    if "warmup_epochs" in row:
        wu_ep = float(row["warmup_epochs"])
        src["warmup_epochs"] = "effective_payload"

    # ----- warmup steps / ratio -----
    wsteps: Optional[int] = None
    src["odcr_warmup_steps"] = "base"
    if "warmup_steps" in row:
        v = int(row["warmup_steps"])
        wsteps = v if v > 0 else None
        src["odcr_warmup_steps"] = "effective_payload"

    wratio: Optional[float] = None
    src["odcr_warmup_ratio"] = "base"
    _require_step3_row_key("warmup_ratio")
    if "warmup_ratio" in row:
        wratio = float(row["warmup_ratio"])
        src["odcr_warmup_ratio"] = "effective_payload"

    # ----- eval batch -----
    eval_bs = int(BASE_TRAINING_DEFAULTS.eval_batch_size)
    src["eval_batch_size"] = "base"
    _require_step3_row_key("eval_batch_size")
    if "eval_batch_size" in row:
        eval_bs = max(1, int(row["eval_batch_size"]))
        src["eval_batch_size"] = "effective_payload"

    # ----- early stop -----
    min_ep = max(1, int(BASE_TRAINING_DEFAULTS.train_min_epochs))
    src["min_epochs"] = "base"
    _require_step3_row_key("min_epochs")
    if "min_epochs" in row:
        min_ep = max(1, int(row["min_epochs"]))
        src["min_epochs"] = "effective_payload"

    esp = max(1, int(BASE_TRAINING_DEFAULTS.train_early_stop_patience))
    src["early_stop_patience"] = "base"
    _require_step3_row_key("early_stop_patience")
    if "early_stop_patience" in row:
        esp = max(1, int(row["early_stop_patience"]))
        src["early_stop_patience"] = "effective_payload"

    if "early_stop_patience_full" in row:
        esp_full = max(1, int(row["early_stop_patience_full"]))
        src["early_stop_patience_full"] = "effective_payload"
    else:
        esp_full = esp
        src["early_stop_patience_full"] = src["early_stop_patience"]

    if "early_stop_patience_loss" in row:
        esp_loss = max(1, int(row["early_stop_patience_loss"]))
        src["early_stop_patience_loss"] = "effective_payload"
    else:
        esp_loss = esp
        src["early_stop_patience_loss"] = src["early_stop_patience"]

    # ----- BLEU samples -----
    b4 = max(64, int(BASE_TRAINING_DEFAULTS.train_bleu4_max_samples))
    src["bleu4_max_samples"] = "base"
    if "bleu4_max_samples" in row:
        b4 = max(64, int(row["bleu4_max_samples"]))
        src["bleu4_max_samples"] = "effective_payload"

    qeval = b4
    src["quick_eval_max_samples"] = "base"
    if "quick_eval_max_samples" in row:
        qeval = max(64, int(row["quick_eval_max_samples"]))
        src["quick_eval_max_samples"] = "effective_payload"

    # ----- full BLEU eval schedule（仅 training_row.full_bleu_eval；旧字段在 resolve 内直接报错）-----
    fe_sched = resolve_full_bleu_eval_from_training_row(row)
    src["full_bleu_eval"] = "effective_payload"

    fb_ds = "inherit"
    src["full_bleu_decode_strategy"] = "base"
    if "full_bleu_decode_strategy" in row:
        fb_ds = parse_full_bleu_decode_strategy(row["full_bleu_decode_strategy"])
        src["full_bleu_decode_strategy"] = "effective_payload"

    ckpt_metric = _normalize_training_checkpoint_metric_yaml(
        row.get("checkpoint_metric", "valid_loss"),
        ctx="effective training_row.checkpoint_metric",
    )
    src["checkpoint_metric"] = "effective_payload" if "checkpoint_metric" in row else "base"
    dual = False

    ema_enabled_v = bool(row.get("ema_enabled", True))
    src["ema_enabled"] = "effective_payload" if "ema_enabled" in row else "base"
    ema_decay_v = float(row.get("ema_decay", 0.999))
    src["ema_decay"] = "effective_payload" if "ema_decay" in row else "base"
    generate_during_train_v = bool(row.get("generate_during_train", False))
    src["generate_during_train"] = "effective_payload" if "generate_during_train" in row else "base"
    from executors.decode_controller import resolve_decode_backend_fallback_policy, resolve_decode_backend_name

    decode_backend_v = resolve_decode_backend_name(row.get("decode_backend", "sdpa_kv_fast"))
    decode_backend_fallback_policy_v = resolve_decode_backend_fallback_policy(
        row.get("decode_backend_fallback_policy", "raise")
    )
    train_time_eval_decode_backend_v = resolve_decode_backend_name(
        row.get("train_time_eval_decode_backend", "sdpa_kv_safe")
    )

    coef_f = 0.0
    src["coef"] = "inactive_step3" if preset_nm == "step3" else "effective_payload"
    if preset_nm != "step3":
        coef_f = float(_coerce_task_param_numeric(row["coef"]))

    # ----- Step5B explainer objective weight. -----
    explainer_loss_weight_f = 0.0
    src["explainer_loss_weight"] = "inactive"
    if preset_nm == "step5":
        if "explainer_loss_weight" not in row or "explainer_loss_weight" not in _payload:
            raise RuntimeError(
                "effective Step5 payload missing explainer_loss_weight; "
                "use step5.train.explainer_loss_weight in configs/odcr.yaml."
            )
        explainer_loss_weight_f = float(_payload["explainer_loss_weight"])
        if not math.isfinite(explainer_loss_weight_f) or explainer_loss_weight_f < 0.0:
            raise ValueError("step5.train.explainer_loss_weight 须为有限非负数")
        src["explainer_loss_weight"] = "step5.train.explainer_loss_weight"

    if getattr(args, "learning_rate", None) is not None:
        src["learning_rate_cli_transport"] = "validated_equal_to_effective_payload"
    if preset_nm != "step3" and getattr(args, "coef", None) is not None:
        src["coef_cli_transport"] = "validated_equal_to_effective_payload"

    # ----- hardware preset 元数据（ODCR_HARDWARE_PRESET 由 runners 注入的 stem；未知 stem 时 blob 为 None）-----
    hardware_preset_nm = (os.environ.get("ODCR_HARDWARE_PRESET") or "").strip() or None

    # ----- max_parallel_cpu / num_proc：已注入 ODCR_HARDWARE_PROFILE_JSON 时仅信任 JSON（单一真相源）-----
    max_par_v: int
    num_proc_v: int
    rp_rt = _active_hardware_preset_slice()
    if rp_rt:
        missing_hw = [k for k in ("max_parallel_cpu", "num_proc") if k not in rp_rt]
        if missing_hw:
            raise RuntimeError(
                "ODCR_HARDWARE_PROFILE_JSON 缺少必需 hardware 字段: "
                + ", ".join(missing_hw)
                + "；请通过 config_resolver 从 configs/odcr.yaml 注入完整 payload。"
            )
        max_par_v = max(1, int(rp_rt["max_parallel_cpu"]))
        src["max_parallel_cpu"] = "hardware_profile_json"
        num_proc_v = max(1, int(rp_rt["num_proc"]))
        num_proc_v = min(num_proc_v, _get_num_cpu())
        src["num_proc"] = "hardware_profile_json"
    else:
        raise RuntimeError(
            "ODCR_HARDWARE_PROFILE_JSON 已设置但无法解析为有效 hardware 切片；"
            "请检查父进程 hardware 预设导出的 JSON。"
        )

    ws = max(int(world_size), 1)
    nw_train = _resolve_ddp_train_num_workers_per_rank_cli(ws, None)
    if rp_rt and (
        "dataloader_num_workers_train" in rp_rt or "dataloader_workers_train_per_rank_cap" in rp_rt
    ):
        src["dataloader_num_workers_train"] = "hardware_profile_json"
    else:
        src["dataloader_num_workers_train"] = "derived"

    dl_valid_base = _resolve_dataloader_num_workers_for_split("valid", None)
    nw_valid = max(1, min(dl_valid_base, nw_train))
    if rp_rt and "dataloader_num_workers_valid" in rp_rt:
        src["dataloader_num_workers_valid"] = "hardware_profile_json"
    else:
        src["dataloader_num_workers_valid"] = "derived"

    nw_test = _resolve_dataloader_num_workers_for_split("test", None)
    if rp_rt and "dataloader_num_workers_test" in rp_rt:
        src["dataloader_num_workers_test"] = "hardware_profile_json"
    else:
        src["dataloader_num_workers_test"] = "derived"

    pf_t = get_dataloader_prefetch_factor(nw_train, split="train")
    pf_v = get_dataloader_prefetch_factor(nw_valid, split="valid")
    pf_test = get_dataloader_prefetch_factor(nw_test, split="test")
    src["dataloader_prefetch_factor_train"] = (
        "hardware_profile_json"
        if rp_rt and "dataloader_prefetch_factor_train" in rp_rt
        else "derived"
    )
    src["dataloader_prefetch_factor_valid"] = (
        "hardware_profile_json"
        if rp_rt and "dataloader_prefetch_factor_valid" in rp_rt
        else "derived"
    )
    src["dataloader_prefetch_factor_test"] = (
        "hardware_profile_json"
        if rp_rt and "dataloader_prefetch_factor_test" in rp_rt
        else "derived"
    )
    if preset_nm == "step3":
        for _hw_req in ("pin_memory", "persistent_workers", "non_blocking_h2d"):
            if _hw_req not in rp_rt:
                raise RuntimeError(f"ODCR_HARDWARE_PROFILE_JSON missing {_hw_req}; Step3 v0 hardware transfer is One-Control.")
    pin_memory_v = bool(rp_rt.get("pin_memory", True))
    persistent_workers_v = bool(rp_rt.get("persistent_workers", True))
    non_blocking_h2d_v = bool(rp_rt.get("non_blocking_h2d", True))
    src["pin_memory"] = "hardware_profile_json" if "pin_memory" in rp_rt else "derived"
    src["persistent_workers"] = "hardware_profile_json" if "persistent_workers" in rp_rt else "derived"
    src["non_blocking_h2d"] = "hardware_profile_json" if "non_blocking_h2d" in rp_rt else "derived"

    _tlm_base = int(max(8, min(512, int(BASE_TRAINING_DEFAULTS.train_label_max_length))))
    src["train_label_max_length"] = "base"
    train_label_max_length_v = _tlm_base
    if "train_label_max_length" in row:
        train_label_max_length_v = int(max(8, min(512, int(row["train_label_max_length"]))))
        src["train_label_max_length"] = "effective_payload"

    train_dynamic_padding_v = True
    src["train_dynamic_padding"] = "base"
    if "train_dynamic_padding" in row:
        train_dynamic_padding_v = bool(row["train_dynamic_padding"])
        src["train_dynamic_padding"] = "effective_payload"

    loss_weight_repeat_ul_v = 0.0
    src["loss_weight_repeat_ul"] = "base"
    if "loss_weight_repeat_ul" in row:
        loss_weight_repeat_ul_v = float(row["loss_weight_repeat_ul"])
        src["loss_weight_repeat_ul"] = "effective_payload"

    loss_weight_terminal_clean_v = 0.0
    src["loss_weight_terminal_clean"] = "base"
    if "loss_weight_terminal_clean" in row:
        loss_weight_terminal_clean_v = float(row["loss_weight_terminal_clean"])
        src["loss_weight_terminal_clean"] = "effective_payload"

    terminal_clean_span_v = 3
    src["terminal_clean_span"] = "base"
    if "terminal_clean_span" in row:
        terminal_clean_span_v = max(1, int(row["terminal_clean_span"]))
        src["terminal_clean_span"] = "effective_payload"

    loss_weight_batch_diversity_v = 0.02
    src["loss_weight_batch_diversity"] = "base"
    if "loss_weight_batch_diversity" in row:
        loss_weight_batch_diversity_v = float(row["loss_weight_batch_diversity"])
        src["loss_weight_batch_diversity"] = "effective_payload"

    lambda_ortho_v = 0.2
    src["lambda_ortho"] = "base"
    if "lambda_ortho" in row:
        lambda_ortho_v = float(row["lambda_ortho"])
        if not math.isfinite(lambda_ortho_v) or lambda_ortho_v < 0.0:
            raise ValueError("effective training_row.lambda_ortho 须为有限非负数")
        src["lambda_ortho"] = "effective_payload"

    lambda_ortho_xcov_v = 1.0
    src["lambda_ortho_xcov"] = "base"
    if "lambda_ortho_xcov" in row:
        lambda_ortho_xcov_v = float(row["lambda_ortho_xcov"])
        if not math.isfinite(lambda_ortho_xcov_v) or lambda_ortho_xcov_v < 0.0:
            raise ValueError("effective training_row.lambda_ortho_xcov 须为有限非负数")
        src["lambda_ortho_xcov"] = "effective_payload"

    lambda_ortho_cos_v = 0.25
    src["lambda_ortho_cos"] = "base"
    if "lambda_ortho_cos" in row:
        lambda_ortho_cos_v = float(row["lambda_ortho_cos"])
        if not math.isfinite(lambda_ortho_cos_v) or lambda_ortho_cos_v < 0.0:
            raise ValueError("effective training_row.lambda_ortho_cos 须为有限非负数")
        src["lambda_ortho_cos"] = "effective_payload"

    lambda_ortho_step5_v = 0.15
    src["lambda_ortho_step5"] = "base"
    if "lambda_ortho_step5" in row:
        lambda_ortho_step5_v = float(row["lambda_ortho_step5"])
        if not math.isfinite(lambda_ortho_step5_v) or lambda_ortho_step5_v < 0.0:
            raise ValueError("effective training_row.lambda_ortho_step5 须为有限非负数")
        src["lambda_ortho_step5"] = "effective_payload"

    retired_step5_weight_keys = sorted(k for k in ("lambda_lci", "lambda_fca") if k in row)
    if retired_step5_weight_keys:
        raise ValueError(
            "effective training_row contains retired Step5 loss keys "
            f"{retired_step5_weight_keys}; use step5.lci.weight / step5.fca.weight via One-Control."
        )

    step5_lci_weight_v = 0.0
    step5_fca_weight_v = 0.0
    src["step5_lci_weight"] = "inactive"
    src["step5_fca_weight"] = "inactive"

    step3_structured_loss_weights_json_v = ""
    step3_loss_semantics_json_v = ""
    method_config_json_v = json.dumps(_payload.get("method") or {}, ensure_ascii=False, sort_keys=True)
    experiment_profile_config_json_v = json.dumps(_payload.get("experiment_profile") or {}, ensure_ascii=False, sort_keys=True)
    experiment_profiles_config_json_v = json.dumps(_payload.get("experiment_profiles") or {}, ensure_ascii=False, sort_keys=True)
    csb_odcr_config_json_v = json.dumps(_payload.get("step3_csb_odcr") or {}, ensure_ascii=False, sort_keys=True)
    csb_contract_config_json_v = json.dumps((_payload.get("step3_csb_odcr") or {}).get("contract") or {}, ensure_ascii=False, sort_keys=True)
    csb_conflict_routing_config_json_v = json.dumps((_payload.get("step3_csb_odcr") or {}).get("conflict_routing") or {}, ensure_ascii=False, sort_keys=True)
    controlled_injection_config_json_v = json.dumps((_payload.get("step3_csb_odcr") or {}).get("controlled_injection") or {}, ensure_ascii=False, sort_keys=True)
    method_obj_v = _payload.get("method") if isinstance(_payload.get("method"), dict) else {}
    experiment_obj_v = _payload.get("experiment_profile") if isinstance(_payload.get("experiment_profile"), dict) else {}
    src["step3_structured_losses"] = "inactive"
    src["step3_loss_semantics"] = "inactive"
    src["method_name"] = "project.method_name"
    src["experiment_profile"] = "step3.experiment_profile"
    src["step3_csb_odcr"] = "step3.csb_odcr"
    if preset_nm == "step3":
        st3_losses = _payload.get("step3_structured_losses")
        if not isinstance(st3_losses, dict):
            raise RuntimeError(
                "effective payload missing step3_structured_losses; "
                "Step3 structured loss weights must come from configs/odcr.yaml."
            )
        step3_structured_loss_weights_json_v = json.dumps(
            st3_losses,
            ensure_ascii=False,
            sort_keys=True,
        )
        src["step3_structured_losses"] = "step3.structured_losses"
        st3_semantics = _payload.get("step3_loss_semantics")
        if not isinstance(st3_semantics, dict):
            raise RuntimeError(
                "effective payload missing step3_loss_semantics; "
                "Step3 active loss semantics must come from configs/odcr.yaml."
            )
        step3_loss_semantics_json_v = json.dumps(
            st3_semantics,
            ensure_ascii=False,
            sort_keys=True,
        )
        src["step3_loss_semantics"] = "step3.loss_semantics"

    step5_innovation_config_json_v = ""
    src["step5_innovation"] = "inactive"
    if preset_nm == "step5":
        st5_innov = _payload.get("step5_innovation")
        if not isinstance(st5_innov, dict):
            raise RuntimeError(
                "effective payload missing step5_innovation; Step5 LCI/UCI/CCV/FCA must come from configs/odcr.yaml."
            )
        step5_innovation_config_json_v = json.dumps(st5_innov, ensure_ascii=False, sort_keys=True)
        step5_lci_weight_v = float(st5_innov.get("lci", {}).get("weight", 0.0))
        step5_fca_weight_v = float(st5_innov.get("fca", {}).get("weight", 0.0))
        if not math.isfinite(step5_lci_weight_v) or step5_lci_weight_v < 0.0:
            raise ValueError("step5.lci.weight 须为有限非负数")
        if not math.isfinite(step5_fca_weight_v) or step5_fca_weight_v < 0.0:
            raise ValueError("step5.fca.weight 须为有限非负数")
        src["step5_lci_weight"] = "step5.lci.weight"
        src["step5_fca_weight"] = "step5.fca.weight"
        src["step5_innovation"] = "effective_payload"

    batch_diversity_warmup_epochs_v = 2
    src["batch_diversity_warmup_epochs"] = "base"
    if "batch_diversity_warmup_epochs" in row:
        batch_diversity_warmup_epochs_v = max(0, int(row["batch_diversity_warmup_epochs"]))
        src["batch_diversity_warmup_epochs"] = "effective_payload"

    batch_diversity_mode_v = "mean_prob_neg_entropy"
    src["batch_diversity_mode"] = "base"
    if "batch_diversity_mode" in row:
        batch_diversity_mode_v = parse_batch_diversity_mode(
            row["batch_diversity_mode"], ctx="effective training_row.batch_diversity_mode"
        )
        src["batch_diversity_mode"] = "effective_payload"

    batch_diversity_eps_v = 1e-8
    src["batch_diversity_eps"] = "base"
    if "batch_diversity_eps" in row:
        batch_diversity_eps_v = float(row["batch_diversity_eps"])
        if batch_diversity_eps_v <= 0.0 or not math.isfinite(batch_diversity_eps_v):
            raise ValueError("batch_diversity_eps 须为有限正数")
        src["batch_diversity_eps"] = "effective_payload"

    batch_diversity_use_ema_v = True
    src["batch_diversity_use_ema"] = "base"
    if "batch_diversity_use_ema" in row:
        batch_diversity_use_ema_v = bool(row["batch_diversity_use_ema"])
        src["batch_diversity_use_ema"] = "effective_payload"

    batch_diversity_ema_decay_v = 0.9
    src["batch_diversity_ema_decay"] = "base"
    if "batch_diversity_ema_decay" in row:
        batch_diversity_ema_decay_v = float(row["batch_diversity_ema_decay"])
        if not math.isfinite(batch_diversity_ema_decay_v) or not (0.0 <= batch_diversity_ema_decay_v < 1.0):
            raise ValueError("batch_diversity_ema_decay 须为 [0,1) 内有限数")
        src["batch_diversity_ema_decay"] = "effective_payload"

    batch_diversity_min_valid_tokens_v = 64
    src["batch_diversity_min_valid_tokens"] = "base"
    if "batch_diversity_min_valid_tokens" in row:
        batch_diversity_min_valid_tokens_v = max(0, int(row["batch_diversity_min_valid_tokens"]))
        src["batch_diversity_min_valid_tokens"] = "effective_payload"

    batch_diversity_loss_clamp_abs_v = 0.2
    src["batch_diversity_loss_clamp_abs"] = "base"
    if "batch_diversity_loss_clamp_abs" in row:
        batch_diversity_loss_clamp_abs_v = float(row["batch_diversity_loss_clamp_abs"])
        if not math.isfinite(batch_diversity_loss_clamp_abs_v) or batch_diversity_loss_clamp_abs_v <= 0.0:
            raise ValueError("batch_diversity_loss_clamp_abs 须为有限正数")
        src["batch_diversity_loss_clamp_abs"] = "effective_payload"

    batch_diversity_ramp_epochs_v = 2
    src["batch_diversity_ramp_epochs"] = "base"
    if "batch_diversity_ramp_epochs" in row:
        batch_diversity_ramp_epochs_v = max(1, int(row["batch_diversity_ramp_epochs"]))
        src["batch_diversity_ramp_epochs"] = "effective_payload"

    batch_diversity_ramp_target_scale_v = 1.0
    src["batch_diversity_ramp_target_scale"] = "base"
    if "batch_diversity_ramp_target_scale" in row:
        batch_diversity_ramp_target_scale_v = float(row["batch_diversity_ramp_target_scale"])
        if not math.isfinite(batch_diversity_ramp_target_scale_v) or batch_diversity_ramp_target_scale_v <= 0.0:
            raise ValueError("batch_diversity_ramp_target_scale 须为有限正数")
        src["batch_diversity_ramp_target_scale"] = "effective_payload"

    batch_diversity_ema_init_mode_v = "uniform"
    src["batch_diversity_ema_init_mode"] = "base"
    if "batch_diversity_ema_init_mode" in row:
        batch_diversity_ema_init_mode_v = parse_batch_diversity_ema_init_mode(
            row["batch_diversity_ema_init_mode"],
            ctx="effective training_row.batch_diversity_ema_init_mode",
        )
        src["batch_diversity_ema_init_mode"] = "effective_payload"

    uncertainty_entropy_eps_v = 1e-8
    src["uncertainty_entropy_eps"] = "base"
    if "uncertainty_entropy_eps" in row:
        uncertainty_entropy_eps_v = float(row["uncertainty_entropy_eps"])
        if not math.isfinite(uncertainty_entropy_eps_v) or uncertainty_entropy_eps_v <= 0.0:
            raise ValueError("uncertainty_entropy_eps 须为有限正数")
        src["uncertainty_entropy_eps"] = "effective_payload"

    uncertainty_high_entropy_threshold_v = 1.0
    src["uncertainty_high_entropy_threshold"] = "base"
    if "uncertainty_high_entropy_threshold" in row:
        uncertainty_high_entropy_threshold_v = float(row["uncertainty_high_entropy_threshold"])
        if not math.isfinite(uncertainty_high_entropy_threshold_v):
            raise ValueError("uncertainty_high_entropy_threshold 须为有限数")
        src["uncertainty_high_entropy_threshold"] = "effective_payload"

    checkpoint_selection_mode_v = "guarded_composite"
    src["checkpoint_selection_mode"] = "base"
    if "checkpoint_selection_mode" in row:
        checkpoint_selection_mode_v = parse_checkpoint_selection_mode(
            row["checkpoint_selection_mode"], ctx="effective training_row.checkpoint_selection_mode"
        )
        src["checkpoint_selection_mode"] = "effective_payload"

    checkpoint_guard_rel_v = 0.005
    src["checkpoint_guard_valid_loss_rel_tol"] = "base"
    if "checkpoint_guard_valid_loss_rel_tol" in row:
        checkpoint_guard_rel_v = float(row["checkpoint_guard_valid_loss_rel_tol"])
        src["checkpoint_guard_valid_loss_rel_tol"] = "effective_payload"

    checkpoint_guard_abs_v = 0.0
    src["checkpoint_guard_valid_loss_abs_tol"] = "base"
    if "checkpoint_guard_valid_loss_abs_tol" in row:
        checkpoint_guard_abs_v = float(row["checkpoint_guard_valid_loss_abs_tol"])
        src["checkpoint_guard_valid_loss_abs_tol"] = "effective_payload"

    cw_b4, cw_rl, cw_m, cw_d1, cw_d2, cw_dirty = 0.32, 0.28, 0.22, 0.05, 0.05, 0.12
    src["checkpoint_composite_w_bleu4"] = "base"
    if "checkpoint_composite_w_bleu4" in row:
        cw_b4 = float(row["checkpoint_composite_w_bleu4"])
        src["checkpoint_composite_w_bleu4"] = "effective_payload"
    src["checkpoint_composite_w_rouge_l"] = "base"
    if "checkpoint_composite_w_rouge_l" in row:
        cw_rl = float(row["checkpoint_composite_w_rouge_l"])
        src["checkpoint_composite_w_rouge_l"] = "effective_payload"
    src["checkpoint_composite_w_meteor"] = "base"
    if "checkpoint_composite_w_meteor" in row:
        cw_m = float(row["checkpoint_composite_w_meteor"])
        src["checkpoint_composite_w_meteor"] = "effective_payload"
    src["checkpoint_composite_w_dist1"] = "base"
    if "checkpoint_composite_w_dist1" in row:
        cw_d1 = float(row["checkpoint_composite_w_dist1"])
        src["checkpoint_composite_w_dist1"] = "effective_payload"
    src["checkpoint_composite_w_dist2"] = "base"
    if "checkpoint_composite_w_dist2" in row:
        cw_d2 = float(row["checkpoint_composite_w_dist2"])
        src["checkpoint_composite_w_dist2"] = "effective_payload"
    src["checkpoint_composite_w_dirty"] = "base"
    if "checkpoint_composite_w_dirty" in row:
        cw_dirty = float(row["checkpoint_composite_w_dirty"])
        src["checkpoint_composite_w_dirty"] = "effective_payload"

    ddp_find_unused_v = False if preset_nm == "step3" else True
    src["ddp_find_unused_parameters"] = "base"
    if "ddp_find_unused_parameters" in row:
        ddp_find_unused_v = bool(row["ddp_find_unused_parameters"])
        src["ddp_find_unused_parameters"] = "effective_payload"
    ddp_find_unused_false_preflight_v = "synthetic_one_batch"
    src["ddp_find_unused_false_preflight"] = "base"
    if "ddp_find_unused_false_preflight" in row:
        ddp_find_unused_false_preflight_v = str(row["ddp_find_unused_false_preflight"]).strip().lower()
        if ddp_find_unused_false_preflight_v not in ("synthetic_one_batch", "fail_fast"):
            raise ValueError("ddp_find_unused_false_preflight 须为 synthetic_one_batch 或 fail_fast")
        if not ddp_find_unused_v and ddp_find_unused_false_preflight_v != "synthetic_one_batch":
            raise ValueError("ddp_find_unused_parameters=false 需要 synthetic_one_batch preflight")
        src["ddp_find_unused_false_preflight"] = "effective_payload"
    ddp_static_graph_v = False
    src["ddp_static_graph"] = "base"
    if "ddp_static_graph" in row:
        ddp_static_graph_v = bool(row["ddp_static_graph"])
        src["ddp_static_graph"] = "effective_payload"
    ddp_graph_safety_preflight_v = True if preset_nm == "step3" else False
    src["ddp_graph_safety_preflight"] = "base"
    if "ddp_graph_safety_preflight" in row:
        ddp_graph_safety_preflight_v = bool(row["ddp_graph_safety_preflight"])
        src["ddp_graph_safety_preflight"] = "effective_payload"

    step5_strict_index_batches_v = False
    src["step5_strict_index_batches"] = "base"
    if "step5_strict_index_batches" in row:
        step5_strict_index_batches_v = bool(row["step5_strict_index_batches"])
        src["step5_strict_index_batches"] = "effective_payload"

    label_smoothing_v = 0.1
    src["label_smoothing"] = "base"
    if "label_smoothing" in row:
        label_smoothing_v = float(row["label_smoothing"])
        if not math.isfinite(label_smoothing_v) or not (0.0 <= label_smoothing_v < 1.0):
            raise ValueError("label_smoothing 须为 [0,1) 内有限数")
        src["label_smoothing"] = "effective_payload"

    repetition_penalty_v = 1.15
    src["repetition_penalty"] = "base"
    if "repetition_penalty" in row:
        repetition_penalty_v = float(row["repetition_penalty"])
        if not math.isfinite(repetition_penalty_v) or repetition_penalty_v <= 0.0:
            raise ValueError("repetition_penalty 须为有限正数")
        src["repetition_penalty"] = "effective_payload"

    generate_temperature_v = 0.8
    src["generate_temperature"] = "base"
    if "generate_temperature" in row:
        generate_temperature_v = max(1e-8, float(row["generate_temperature"]))
        if not math.isfinite(generate_temperature_v):
            raise ValueError("generate_temperature 须为有限正数")
        src["generate_temperature"] = "effective_payload"

    max_explanation_length_v = 25
    src["max_explanation_length"] = "base"
    if "max_explanation_length" in row:
        max_explanation_length_v = max(1, int(row["max_explanation_length"]))
        src["max_explanation_length"] = "effective_payload"

    gap_threshold_v = 0.35
    src["gap_threshold"] = "base"
    if "gap_threshold" in row:
        gap_threshold_v = float(row["gap_threshold"])
        if not math.isfinite(gap_threshold_v):
            raise ValueError("gap_threshold 须为有限数")
        src["gap_threshold"] = "effective_payload"

    prefix_greedy_steps_v = 4
    src["prefix_greedy_steps"] = "base"
    if "prefix_greedy_steps" in row:
        prefix_greedy_steps_v = max(0, int(row["prefix_greedy_steps"]))
        src["prefix_greedy_steps"] = "effective_payload"

    decode_top_k_v = 5
    src["decode_top_k"] = "base"
    if "decode_top_k" in row:
        decode_top_k_v = max(1, int(row["decode_top_k"]))
        src["decode_top_k"] = "effective_payload"

    emsize_v = int(get_odcr_embed_dim())
    src["emsize"] = "odcr_embed_dim"
    if "emsize" in row:
        emsize_v = max(1, int(row["emsize"]))
        src["emsize"] = "effective_payload"
    _reject_conflicting_child_cli("emsize", emsize_v, "training_row.emsize/env.embed_dim", integer=True)
    if getattr(args, "emsize", None) is not None:
        src["emsize_cli_transport"] = "validated_equal_to_effective_payload"

    if preset_nm == "step5":
        missing_model = [key for key in ("emsize", "nlayers", "nhead", "nhid", "dropout") if key not in row]
        if missing_model:
            raise RuntimeError(
                "effective Step5 payload missing architecture fields "
                f"{missing_model}; they must be injected from step5.model."
            )
    nlayers_v = max(1, int(row.get("nlayers", 2)))
    src["nlayers"] = "step5.model" if preset_nm == "step5" else ("effective_payload" if "nlayers" in row else "base")
    nhead_v = max(1, int(row.get("nhead", 2)))
    src["nhead"] = "step5.model" if preset_nm == "step5" else ("effective_payload" if "nhead" in row else "base")
    nhid_v = max(1, int(row.get("nhid", 2048)))
    src["nhid"] = "step5.model" if preset_nm == "step5" else ("effective_payload" if "nhid" in row else "base")
    dropout_v = float(row.get("dropout", 0.2))
    if not math.isfinite(dropout_v) or not (0.0 <= dropout_v <= 1.0):
        raise ValueError("effective training_row.dropout 须为 [0,1] 内有限数")
    src["dropout"] = "step5.model" if preset_nm == "step5" else ("effective_payload" if "dropout" in row else "base")
    _reject_conflicting_child_cli("nlayers", nlayers_v, "training_row.nlayers/step5.model.nlayers", integer=True)
    _reject_conflicting_child_cli("nhead", nhead_v, "training_row.nhead/step5.model.nhead", integer=True)
    _reject_conflicting_child_cli("nhid", nhid_v, "training_row.nhid/step5.model.nhid", integer=True)
    _reject_conflicting_child_cli("dropout", dropout_v, "training_row.dropout/step5.model.dropout")

    if preset_nm == "step5":
        missing_lora = [
            key
            for key in ("train_mode", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules")
            if key not in row
        ]
        if missing_lora:
            raise RuntimeError(
                "effective Step5 payload missing native LoRA runtime fields "
                f"{missing_lora}; they must be injected from step5.ccv.native_lora."
            )
    train_mode_v = parse_train_mode(
        row.get("train_mode", "full"), ctx="effective training_row.train_mode"
    )
    src["train_mode"] = "step5.ccv.native_lora" if preset_nm == "step5" else ("effective_payload" if "train_mode" in row else "base")
    if "train_precision" not in row:
        if preset_nm == "step3":
            raise RuntimeError(
                "effective Step3 payload missing train_precision; "
                "use step3.train.backend.train_precision in configs/odcr.yaml."
            )
        train_precision_raw = "bf16"
        src["train_precision"] = "base"
    else:
        train_precision_raw = row["train_precision"]
        src["train_precision"] = "effective_payload"
    train_precision_v = parse_train_precision(
        train_precision_raw, ctx="effective training_row.train_precision"
    )
    if preset_nm == "step3" and train_precision_v != "bf16":
        raise RuntimeError("effective Step3 train_precision must be bf16.")
    allow_tf32_v = bool(row.get("allow_tf32", False))
    amp_autocast_v = bool(row.get("amp_autocast", False))
    grad_scaler_v = bool(row.get("grad_scaler", False))
    src["allow_tf32"] = "effective_payload" if "allow_tf32" in row else "base"
    src["amp_autocast"] = "effective_payload" if "amp_autocast" in row else "base"
    src["grad_scaler"] = "effective_payload" if "grad_scaler" in row else "base"
    if preset_nm == "step3":
        for key in ("allow_tf32", "amp_autocast", "grad_scaler"):
            _require_step3_row_key(key)
        if not allow_tf32_v or not amp_autocast_v or grad_scaler_v:
            raise RuntimeError("Step3 v0 precision backend requires allow_tf32=true, amp_autocast=true, grad_scaler=false.")

    def _payload_obj(key: str) -> dict[str, Any]:
        obj = _payload.get(key)
        if preset_nm == "step3" and not isinstance(obj, dict):
            raise RuntimeError(f"effective Step3 payload missing {key}; configs/odcr.yaml must own this control.")
        return dict(obj or {})

    optimizer_config_v = _payload_obj("step3_optimizer")
    precision_config_v = _payload_obj("step3_precision")
    tokenizer_config_v = _payload_obj("step3_tokenizer")
    evidence_config_v = _payload_obj("step3_evidence")
    scheduler_config_v = _payload_obj("step3_scheduler")
    valid_batch_config_v = _payload_obj("step3_eval")
    scenario_profile_v = _payload_obj("step3_scenario_profile")
    task_profile_config_v = _payload_obj("step3_task_profile")
    backup_profiles_config_v = _payload_obj("step3_backup_profiles")
    exploration_profiles_config_v = _payload_obj("step3_exploration_profiles")
    worker_profiles_config_v = _payload_obj("step3_worker_profiles")
    prefetcher_config_v = _payload_obj("step3_prefetcher")
    checkpoint_policy_config_v = _payload_obj("step3_checkpoint_policy")
    quality_gate_config_v = _payload_obj("step3_quality_gate")
    grad_finite_config_v = _payload_obj("step3_grad_finite")
    numerical_stability_config_v = _payload_obj("step3_numerical_stability")
    diagnostic_eval_config_v = _payload_obj("step3_diagnostic_eval")
    cross_rank_structured_gather_config_v = _payload_obj("step3_cross_rank_structured_gather")
    memory_config_v = _payload_obj("step3_memory")
    timing_config_v = _payload_obj("step3_timing")
    performance_candidates_config_v = _payload_obj("step3_performance_candidates")
    cache_policy_config_v = _payload_obj("step3_cache_policy")
    objective_drift_config_v = _payload_obj("step3_objective_drift")
    recovery_config_v = _payload_obj("step3_recovery")
    phase_loss_schedule_config_v = _payload_obj("step3_phase_loss_schedule")
    conflict_aware_config_v = _payload_obj("step3_conflict_aware")
    loss_gradient_conflict_probe_config_v = _payload_obj("step3_loss_gradient_conflict_probe")
    adapter_gating_config_v = _payload_obj("step3_adapter_gating")
    paper_candidate_selection_config_v = _payload_obj("step3_paper_candidate_selection")
    checkpoint_averaging_config_v = _payload_obj("step3_checkpoint_averaging")
    tokenizer_max_length_v = int(row.get("tokenizer_max_length", 0))
    evidence_max_length_v = int(row.get("evidence_max_length", 0))
    max_grad_norm_v = float(row.get("max_grad_norm", 0.0))
    validate_every_epochs_v = int(row.get("validate_every_epochs", 1))
    valid_batch_size_v = int(row.get("valid_batch_size", eval_bs))
    valid_micro_batch_size_v = int(row.get("valid_micro_batch_size", max(1, eval_bs // max(1, world_size))))
    src["optimizer"] = "step3.optimizer" if preset_nm == "step3" else "inactive"
    src["tokenizer_max_length"] = "effective_payload" if "tokenizer_max_length" in row else "inactive"
    src["evidence_max_length"] = "effective_payload" if "evidence_max_length" in row else "inactive"
    src["max_grad_norm"] = "effective_payload" if "max_grad_norm" in row else "inactive"
    src["validate_every_epochs"] = "effective_payload" if "validate_every_epochs" in row else "inactive"
    src["valid_batch_size"] = "effective_payload" if "valid_batch_size" in row else "inactive"
    src["valid_micro_batch_size"] = "effective_payload" if "valid_micro_batch_size" in row else "inactive"
    if preset_nm == "step3":
        for key in (
            "optimizer",
            "tokenizer_max_length",
            "evidence_max_length",
            "max_grad_norm",
            "validate_every_epochs",
            "valid_batch_size",
            "valid_micro_batch_size",
        ):
            _require_step3_row_key(key)
        if str(optimizer_config_v.get("name") or "").lower() != "adamw":
            raise RuntimeError("Step3 optimizer must be AdamW in effective payload.")
        if max_grad_norm_v <= 0.0:
            raise RuntimeError("Step3 max_grad_norm must be positive.")
        src["task_profile_id"] = "effective_payload"
        src["profile_isolation_hash"] = "effective_payload"
    per_device_eval_v = max(1, int(row.get("per_device_eval_batch_size", 2)))
    src["per_device_eval_batch_size"] = "effective_payload" if "per_device_eval_batch_size" in row else "base"
    lora_r_v = max(1, int(row.get("lora_r", 16)))
    src["lora_r"] = "step5.ccv.native_lora" if preset_nm == "step5" else ("effective_payload" if "lora_r" in row else "base")
    lora_alpha_v = float(row.get("lora_alpha", 32.0))
    src["lora_alpha"] = "step5.ccv.native_lora" if preset_nm == "step5" else ("effective_payload" if "lora_alpha" in row else "base")
    lora_dropout_v = float(row.get("lora_dropout", 0.05))
    src["lora_dropout"] = "step5.ccv.native_lora" if preset_nm == "step5" else ("effective_payload" if "lora_dropout" in row else "base")
    _ltm = row.get("lora_target_modules")
    if _ltm is None:
        lora_targets_v: Tuple[str, ...] = ()
        src["lora_target_modules"] = "base"
    else:
        if not isinstance(_ltm, list):
            raise TypeError("effective training_row.lora_target_modules 须为 str 列表或省略")
        lora_targets_v = tuple(str(x).strip() for x in _ltm if str(x).strip())
        src["lora_target_modules"] = "step5.ccv.native_lora" if preset_nm == "step5" else "effective_payload"

    sources_tuple = tuple(sorted(src.items()))

    return FinalTrainingConfig(
        task_idx=tid,
        auxiliary=auxiliary,
        target=target,
        scenario=str(_payload.get("scenario") or row.get("scenario") or "legacy_scenario"),
        direction=str(_payload.get("direction") or row.get("direction") or "unspecified"),
        task_profile_id=str(_payload.get("task_profile_id") or row.get("task_profile_id") or ""),
        task_profile_key=str(_payload.get("task_profile_key") or row.get("task_profile_key") or ""),
        profile_isolation_hash=str(_payload.get("profile_isolation_hash") or row.get("profile_isolation_hash") or ""),
        preset_name=preset_nm,
        world_size=int(world_size),
        sources=sources_tuple,
        learning_rate=initial_f,
        scheduler_initial_lr=initial_f,
        initial_lr=initial_f,
        epochs=ep,
        max_epochs=int(row.get("max_epochs", ep)),
        validate_every_epochs=int(validate_every_epochs_v),
        max_grad_norm=float(max_grad_norm_v),
        tokenizer_max_length=int(tokenizer_max_length_v),
        evidence_max_length=int(evidence_max_length_v),
        valid_batch_size=int(valid_batch_size_v),
        valid_micro_batch_size=int(valid_micro_batch_size_v),
        train_batch_size=G,
        global_batch_size=G,
        batch_size_global=G,
        batch_size=P,
        per_device_train_batch_size=P,
        per_gpu_batch_size=P,
        effective_global_batch_size=eff,
        batch_semantics_version=NO_ACCUM_BATCH_SEMANTICS_VERSION,
        grad_accum_removed=True,
        num_proc=num_proc_v,
        max_parallel_cpu=max_par_v,
        hardware_preset_name=hardware_preset_nm,
        dataloader_num_workers_train=nw_train,
        dataloader_num_workers_valid=nw_valid,
        dataloader_num_workers_test=nw_test,
        dataloader_prefetch_factor_train=pf_t,
        dataloader_prefetch_factor_valid=pf_v,
        dataloader_prefetch_factor_test=pf_test,
        pin_memory=bool(pin_memory_v),
        persistent_workers=bool(persistent_workers_v),
        non_blocking_h2d=bool(non_blocking_h2d_v),
        min_lr_ratio=min_lr,
        lr_scheduler=lr_sched,
        scheduler_type=lr_sched,
        warmup_epochs=wu_ep,
        odcr_warmup_steps=wsteps,
        odcr_warmup_ratio=wratio,
        optimizer_config_json=json.dumps(optimizer_config_v, ensure_ascii=False, sort_keys=True),
        precision_config_json=json.dumps(precision_config_v, ensure_ascii=False, sort_keys=True),
        tokenizer_config_json=json.dumps(tokenizer_config_v, ensure_ascii=False, sort_keys=True),
        evidence_config_json=json.dumps(evidence_config_v, ensure_ascii=False, sort_keys=True),
        scheduler_config_json=json.dumps(scheduler_config_v, ensure_ascii=False, sort_keys=True),
        valid_batch_config_json=json.dumps(valid_batch_config_v, ensure_ascii=False, sort_keys=True),
        scenario_profile_json=json.dumps(scenario_profile_v, ensure_ascii=False, sort_keys=True),
        task_profile_config_json=json.dumps(task_profile_config_v, ensure_ascii=False, sort_keys=True),
        backup_profiles_config_json=json.dumps(backup_profiles_config_v, ensure_ascii=False, sort_keys=True),
        exploration_profiles_config_json=json.dumps(exploration_profiles_config_v, ensure_ascii=False, sort_keys=True),
        worker_profiles_config_json=json.dumps(worker_profiles_config_v, ensure_ascii=False, sort_keys=True),
        prefetcher_config_json=json.dumps(prefetcher_config_v, ensure_ascii=False, sort_keys=True),
        checkpoint_policy_config_json=json.dumps(checkpoint_policy_config_v, ensure_ascii=False, sort_keys=True),
        quality_gate_config_json=json.dumps(quality_gate_config_v, ensure_ascii=False, sort_keys=True),
        grad_finite_config_json=json.dumps(grad_finite_config_v, ensure_ascii=False, sort_keys=True),
        numerical_stability_config_json=json.dumps(numerical_stability_config_v, ensure_ascii=False, sort_keys=True),
        diagnostic_eval_config_json=json.dumps(diagnostic_eval_config_v, ensure_ascii=False, sort_keys=True),
        cross_rank_structured_gather_config_json=json.dumps(cross_rank_structured_gather_config_v, ensure_ascii=False, sort_keys=True),
        memory_config_json=json.dumps(memory_config_v, ensure_ascii=False, sort_keys=True),
        timing_config_json=json.dumps(timing_config_v, ensure_ascii=False, sort_keys=True),
        performance_candidates_config_json=json.dumps(performance_candidates_config_v, ensure_ascii=False, sort_keys=True),
        cache_policy_config_json=json.dumps(cache_policy_config_v, ensure_ascii=False, sort_keys=True),
        objective_drift_config_json=json.dumps(objective_drift_config_v, ensure_ascii=False, sort_keys=True),
        recovery_config_json=json.dumps(recovery_config_v, ensure_ascii=False, sort_keys=True),
        phase_loss_schedule_config_json=json.dumps(phase_loss_schedule_config_v, ensure_ascii=False, sort_keys=True),
        conflict_aware_config_json=json.dumps(conflict_aware_config_v, ensure_ascii=False, sort_keys=True),
        loss_gradient_conflict_probe_config_json=json.dumps(loss_gradient_conflict_probe_config_v, ensure_ascii=False, sort_keys=True),
        adapter_gating_config_json=json.dumps(adapter_gating_config_v, ensure_ascii=False, sort_keys=True),
        paper_candidate_selection_config_json=json.dumps(paper_candidate_selection_config_v, ensure_ascii=False, sort_keys=True),
        checkpoint_averaging_config_json=json.dumps(checkpoint_averaging_config_v, ensure_ascii=False, sort_keys=True),
        eval_batch_size=eval_bs,
        min_epochs=min_ep,
        train_min_epochs=min_ep,
        early_stop_patience=esp,
        early_stop_patience_full=esp_full,
        early_stop_patience_loss=esp_loss,
        full_bleu_eval_resolved=fe_sched,
        checkpoint_metric=ckpt_metric,
        dual_bleu_eval=dual,
        bleu4_max_samples=b4,
        quick_eval_max_samples=qeval,
        coef=coef_f,
        explainer_loss_weight=explainer_loss_weight_f,
        full_bleu_decode_strategy=fb_ds,
        ema_enabled=bool(ema_enabled_v),
        ema_decay=float(ema_decay_v),
        generate_during_train=bool(generate_during_train_v),
        decode_backend=str(decode_backend_v),
        decode_backend_fallback_policy=str(decode_backend_fallback_policy_v),
        train_time_eval_decode_backend=str(train_time_eval_decode_backend_v),
        train_label_max_length=int(train_label_max_length_v),
        train_dynamic_padding=bool(train_dynamic_padding_v),
        train_padding_strategy=("dynamic_batch" if bool(train_dynamic_padding_v) else "fixed_max_length"),
        loss_weight_repeat_ul=float(loss_weight_repeat_ul_v),
        loss_weight_terminal_clean=float(loss_weight_terminal_clean_v),
        terminal_clean_span=int(terminal_clean_span_v),
        loss_weight_batch_diversity=float(loss_weight_batch_diversity_v),
        batch_diversity_warmup_epochs=int(batch_diversity_warmup_epochs_v),
        batch_diversity_mode=str(batch_diversity_mode_v),
        batch_diversity_eps=float(batch_diversity_eps_v),
        batch_diversity_use_ema=bool(batch_diversity_use_ema_v),
        batch_diversity_ema_decay=float(batch_diversity_ema_decay_v),
        batch_diversity_min_valid_tokens=int(batch_diversity_min_valid_tokens_v),
        batch_diversity_loss_clamp_abs=float(batch_diversity_loss_clamp_abs_v),
        batch_diversity_ramp_epochs=int(batch_diversity_ramp_epochs_v),
        batch_diversity_ramp_target_scale=float(batch_diversity_ramp_target_scale_v),
        batch_diversity_ema_init_mode=str(batch_diversity_ema_init_mode_v),
        lambda_ortho=float(lambda_ortho_v),
        lambda_ortho_xcov=float(lambda_ortho_xcov_v),
        lambda_ortho_cos=float(lambda_ortho_cos_v),
        lambda_ortho_step5=float(lambda_ortho_step5_v),
        step5_lci_weight=float(step5_lci_weight_v),
        step5_fca_weight=float(step5_fca_weight_v),
        method_name=str(method_obj_v.get("method_name") or "CSB-ODCR"),
        method_full_name=str(
            method_obj_v.get("method_full_name")
            or "CSB-ODCR: Causal Structure Bottleneck for Orthogonal Disentangled Counterfactual Recommendation"
        ),
        method_family=str(method_obj_v.get("method_family") or "csb_odcr"),
        experiment_profile=str(experiment_obj_v.get("name") or _payload.get("ablation_profile") or "csb_odcr_sidecar_stable"),
        ablation_profile=str(experiment_obj_v.get("name") or _payload.get("ablation_profile") or "csb_odcr_sidecar_stable"),
        method_config_json=str(method_config_json_v),
        experiment_profile_config_json=str(experiment_profile_config_json_v),
        experiment_profiles_config_json=str(experiment_profiles_config_json_v),
        csb_odcr_config_json=str(csb_odcr_config_json_v),
        csb_contract_config_json=str(csb_contract_config_json_v),
        csb_conflict_routing_config_json=str(csb_conflict_routing_config_json_v),
        controlled_injection_config_json=str(controlled_injection_config_json_v),
        step5_innovation_config_json=str(step5_innovation_config_json_v),
        step3_structured_loss_weights_json=str(step3_structured_loss_weights_json_v),
        step3_loss_semantics_json=str(step3_loss_semantics_json_v),
        uncertainty_entropy_eps=float(uncertainty_entropy_eps_v),
        uncertainty_high_entropy_threshold=float(uncertainty_high_entropy_threshold_v),
        checkpoint_selection_mode=str(checkpoint_selection_mode_v),
        checkpoint_guard_valid_loss_rel_tol=float(checkpoint_guard_rel_v),
        checkpoint_guard_valid_loss_abs_tol=float(checkpoint_guard_abs_v),
        checkpoint_composite_w_bleu4=float(cw_b4),
        checkpoint_composite_w_rouge_l=float(cw_rl),
        checkpoint_composite_w_meteor=float(cw_m),
        checkpoint_composite_w_dist1=float(cw_d1),
        checkpoint_composite_w_dist2=float(cw_d2),
        checkpoint_composite_w_dirty=float(cw_dirty),
        ddp_find_unused_parameters=ddp_find_unused_v,
        ddp_find_unused_false_preflight=str(ddp_find_unused_false_preflight_v),
        ddp_static_graph=bool(ddp_static_graph_v),
        ddp_graph_safety_preflight=bool(ddp_graph_safety_preflight_v),
        step5_strict_index_batches=bool(step5_strict_index_batches_v),
        label_smoothing=float(label_smoothing_v),
        repetition_penalty=float(repetition_penalty_v),
        generate_temperature=float(generate_temperature_v),
        max_explanation_length=int(max_explanation_length_v),
        gap_threshold=float(gap_threshold_v),
        prefix_greedy_steps=int(prefix_greedy_steps_v),
        decode_top_k=int(decode_top_k_v),
        emsize=int(emsize_v),
        nlayers=int(nlayers_v),
        nhead=int(nhead_v),
        nhid=int(nhid_v),
        dropout=float(dropout_v),
        train_mode=str(train_mode_v),
        train_precision=str(train_precision_v),
        allow_tf32=bool(allow_tf32_v),
        amp_autocast=bool(amp_autocast_v),
        grad_scaler=bool(grad_scaler_v),
        per_device_eval_batch_size=int(per_device_eval_v),
        lora_r=int(lora_r_v),
        lora_alpha=float(lora_alpha_v),
        lora_dropout=float(lora_dropout_v),
        lora_target_modules=lora_targets_v,
    )


def get_train_batch_size(task_idx: Optional[int] = None) -> int:
    """
    全局训练 batch G；供**无 odcr CLI** 的极薄辅助场景。
    顺序：BASE。One-Control 主链的覆盖只能来自 configs/odcr.yaml 或 CLI --set。
    ``task_idx`` 保留以兼容旧调用方，**不**再参与解析。
    """
    _ = task_idx
    return int(BASE_TRAINING_DEFAULTS.train_batch_size)


def get_epochs(task_idx: Optional[int] = None) -> int:
    """
    训练轮数；供**无 odcr CLI** 的极薄辅助场景。
    顺序：BASE。One-Control 主链的覆盖只能来自 configs/odcr.yaml 或 CLI --set。
    ``task_idx`` 保留以兼容旧调用方，**不**再参与解析。
    """
    _ = task_idx
    return int(BASE_TRAINING_DEFAULTS.epochs)


def __getattr__(name: str) -> Any:
    """惰性解析：``from config import num_proc`` / ``eval_batch_size`` 等。"""
    if name == "num_proc":
        return get_num_proc()
    if name == "eval_batch_size":
        return get_eval_batch_size()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
