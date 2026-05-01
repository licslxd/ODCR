# -*- coding: utf-8 -*-
"""
ODCR 离线运行路径配置（新版）

- 文本/句向量权重加载：调用方须使用 ``from_pretrained(..., local_files_only=True)``，
  并先经本模块 ``require_step5_text_model_dir`` / ``require_sentence_embed_model_dir`` 做目录 fail-fast；
  不得依赖「仅环境变量 HF_HUB_OFFLINE」作为唯一防联网手段。
- 项目根：ODCR_ROOT（默认 code 上级）。
- 数据根：由 ``configs/odcr.yaml`` 的 ``project.data_dir`` / ``project.merged_dir`` 解析，
  父进程仅以 ``ODCR_RESOLVED_*`` 传给 torchrun 子进程。
- 当前阶段产物根目录：须设置 **ODCR_STAGE_RUN_DIR**（由 ``python code/odcr.py`` 在 torchrun 前注入），
  对应 ``runs/task{T}/vN/train/step3|step4|step5/<run>/`` 等。
- Step4 另可选 **ODCR_STEP3_RUN_DIR**：仅 step4 runner 加载 Step3 权重时指向 ``train/step3/<from-run>/``（CSV 与 partial 仍写入 ``ODCR_STAGE_RUN_DIR``）。
- HF datasets 缓存根：固定为 ``<repo>/cache/task{T}/hf``，不读取用户环境变量。

路径常量请使用本模块显式 ``get_*()``；不再提供模块级惰性属性或 ``__getattr__`` 动态导出。
"""
from __future__ import annotations

import os
import json
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 与离线镜像目录名对应的 HuggingFace 模型 id（写入 index_contract / manifest，非运行时下载）
DEFAULT_SENTENCE_EMBED_MODEL_ID = "BAAI/bge-large-en-v1.5"
DEFAULT_STEP5_TEXT_MODEL_ID = "google/flan-t5-xl"


def get_odcr_root() -> str:
    """项目根目录；运行时读取 ODCR_ROOT，默认 code 的上一级。"""
    env = os.environ.get("ODCR_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _repo_path(raw: object) -> str:
    text = str(raw or "").strip()
    if not text:
        raise RuntimeError("ODCR One-Control path is empty; check configs/odcr.yaml")
    p = Path(os.path.expanduser(text))
    if not p.is_absolute():
        p = Path(get_odcr_root()) / p
    return str(p.resolve())


def _runtime_roots_from_payload() -> dict[str, object]:
    raw = (os.environ.get("ODCR_EFFECTIVE_TRAINING_PAYLOAD_JSON") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    roots = payload.get("runtime_roots") if isinstance(payload, dict) else None
    return dict(roots) if isinstance(roots, dict) else {}


def _runtime_roots_from_yaml() -> dict[str, object]:
    cfg_path = Path(get_odcr_root()) / "configs" / "odcr.yaml"
    if not cfg_path.is_file():
        raise RuntimeError(
            "缺少 configs/odcr.yaml；全局路径/模型路径/embed_dim 必须来自 One-Control 配置或父进程 ODCR_RESOLVED_* 注入。"
        )
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/odcr.yaml for One-Control paths") from exc
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("configs/odcr.yaml 根必须为 object")
    project = raw.get("project") if isinstance(raw.get("project"), dict) else {}
    env = raw.get("env") if isinstance(raw.get("env"), dict) else {}
    return {
        "data_dir": project.get("data_dir"),
        "merged_dir": project.get("merged_dir"),
        "models_dir": env.get("models_dir"),
        "step5_text_model": env.get("step5_text_model"),
        "sentence_embed_model": env.get("sentence_embed_model"),
        "embed_dim": env.get("embed_dim"),
    }


def _resolved_runtime_value(key: str, resolved_env: str, legacy_env: str | None = None) -> str:
    injected = (os.environ.get(resolved_env) or "").strip()
    if injected:
        value = _repo_path(injected)
    else:
        payload = _runtime_roots_from_payload()
        if key in payload and str(payload.get(key) or "").strip():
            value = _repo_path(payload[key])
        else:
            value = _repo_path(_runtime_roots_from_yaml().get(key))
    if legacy_env:
        legacy = (os.environ.get(legacy_env) or "").strip()
        if legacy:
            legacy_value = _repo_path(legacy)
            if legacy_value != value:
                raise RuntimeError(
                    f"{legacy_env}={legacy_value} conflicts with One-Control {key}={value}; "
                    "旧 ODCR_* 环境变量不得覆盖 configs/odcr.yaml。"
                )
    return value


def get_resolved_embed_dim() -> int:
    injected = (os.environ.get("ODCR_RESOLVED_EMBED_DIM") or "").strip()
    if injected:
        value = int(injected)
    else:
        payload = _runtime_roots_from_payload()
        if str(payload.get("embed_dim") or "").strip():
            value = int(payload["embed_dim"])
        else:
            value = int(_runtime_roots_from_yaml().get("embed_dim"))
    legacy = (os.environ.get("ODCR_EMBED_DIM") or "").strip()
    if legacy and int(legacy) != int(value):
        raise RuntimeError(
            f"ODCR_EMBED_DIM={int(legacy)} conflicts with One-Control env.embed_dim={int(value)}; "
            "旧 ODCR_EMBED_DIM 只能与 resolver 注入值一致，不能作为用户侧优先来源。"
        )
    if value <= 0:
        raise RuntimeError("env.embed_dim must be a positive integer in configs/odcr.yaml")
    return int(value)


def get_models_dir() -> str:
    return _resolved_runtime_value("models_dir", "ODCR_RESOLVED_MODELS_DIR", "ODCR_MODELS_DIR")


def get_step5_text_model_dir() -> str:
    """Step5/Step3/Step4 共用的 T5Tokenizer 本地目录（默认：google/flan-t5-xl 离线镜像名）。"""
    return _resolved_runtime_value(
        "step5_text_model",
        "ODCR_RESOLVED_STEP5_TEXT_MODEL",
        "ODCR_STEP5_TEXT_MODEL",
    )


def require_step5_text_model_dir() -> str:
    p = get_step5_text_model_dir()
    if not os.path.isdir(p):
        raise FileNotFoundError(
            "离线 Step5 文本模型（T5Tokenizer）目录缺失：google flan-t5-xl 本地目录不存在。\n"
            f"  期望路径: {p}\n"
            "  请将 Hugging Face 格式的 google/flan-t5-xl 放入 configs/odcr.yaml env.step5_text_model 指向的目录。\n"
            "  禁止静默回退到 t5-small 或从 Hugging Face Hub 联网下载。"
        )
    return p


def get_sentence_embed_model_dir() -> str:
    """Sentence / domain 语义嵌入模型本地目录（默认：BAAI/bge-large-en-v1.5 离线镜像名）。"""
    return _resolved_runtime_value(
        "sentence_embed_model",
        "ODCR_RESOLVED_SENTENCE_EMBED_MODEL",
        "ODCR_SENTENCE_EMBED_MODEL",
    )


def require_sentence_embed_model_dir() -> str:
    p = get_sentence_embed_model_dir()
    if not os.path.isdir(p):
        raise FileNotFoundError(
            "离线句向量模型目录缺失：BAAI bge-large-en-v1.5 本地目录不存在。\n"
            f"  期望路径: {p}\n"
            "  请将 Hugging Face 格式的 BAAI/bge-large-en-v1.5 放入 configs/odcr.yaml env.sentence_embed_model 指向的目录。\n"
            "  禁止静默回退到 mpnet 或从 Hugging Face Hub 联网下载。"
        )
    return p


def get_t5_small_dir() -> str:
    """历史名：与 :func:`get_step5_text_model_dir` 相同（当前主线为 flan-t5-xl 目录）。"""
    return get_step5_text_model_dir()


def get_mpnet_dir() -> str:
    """历史名：与 :func:`get_sentence_embed_model_dir` 相同（当前主线为 BGE-large）。"""
    return get_sentence_embed_model_dir()


def get_meteor_cache_dir() -> str:
    """METEOR/evaluate 本地缓存目录；默认 ``<repo>/artifacts/models/evaluate_meteor``。"""
    return os.path.join(get_models_dir(), "evaluate_meteor")


def get_meteor_metric_module_dir() -> str:
    """evaluate meteor 本地脚本镜像根目录；默认 ``<repo>/artifacts/models/hf_cache/.../evaluate-metric--meteor``。"""
    return os.path.join(
        get_models_dir(),
        "hf_cache",
        "modules",
        "evaluate_modules",
        "metrics",
        "evaluate-metric--meteor",
    )


def get_data_dir() -> str:
    return _resolved_runtime_value("data_dir", "ODCR_RESOLVED_DATA_DIR", "ODCR_DATA_DIR")


def get_merged_data_dir() -> str:
    return _resolved_runtime_value("merged_dir", "ODCR_RESOLVED_MERGED_DIR", "ODCR_MERGED_DATA_DIR")


def get_stage_run_dir(_task_idx: int | None = None) -> str:
    """当前 **stage run 根目录**（环境变量 ``ODCR_STAGE_RUN_DIR``）。

    权重与当次 CSV 等产物与此目录对齐。``_task_idx`` 仅为与旧调用形态对齐，不参与解析。
    """
    _ = _task_idx
    stage = os.environ.get("ODCR_STAGE_RUN_DIR", "").strip()
    if not stage:
        raise RuntimeError(
            "ODCR_STAGE_RUN_DIR 未设置。请使用仓库根目录的: python code/odcr.py <子命令> … 启动训练/评测。"
        )
    return os.path.abspath(stage)


def get_hf_cache_root(task_idx: int) -> str:
    """HF datasets 缓存根目录；固定在 cache/task{T}/hf 以便统一清理与 lineage 审计。"""
    return os.path.abspath(os.path.join(get_odcr_root(), "cache", f"task{int(task_idx)}", "hf"))


CODE_DIR = _SCRIPT_DIR
DEFAULT_MIRROR_LOG = ""


def get_nltk_data_dir() -> str:
    """NLTK 数据根目录（punkt / wordnet / omw-1.4 等）；默认 ``<repo>/artifacts/nltk_data``。"""
    env = os.environ.get("ODCR_NLTK_DATA", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(get_odcr_root(), "artifacts", "nltk_data")


def require_nltk_data_dir() -> str:
    """评测链使用的 NLTK 资源目录；缺失时 fail-fast，不尝试联网下载。"""
    p = get_nltk_data_dir()
    if not os.path.isdir(p):
        raise FileNotFoundError(
            "NLTK 本地数据目录缺失（METEOR / word_tokenize 等需要）。\n"
            f"  期望路径: {p}\n"
            "  可通过环境变量 ODCR_NLTK_DATA 指向已有 nltk_data 目录。\n"
            "  禁止依赖运行时 nltk.download 联网拉取。"
        )
    return os.path.abspath(p)


def _mirror_enabled():
    return False


def append_log_dual(primary_log_file, text, mirror=None):
    """Write only the primary run-meta log; fallback mirror logs are retired."""
    _ = mirror
    paths = []
    if primary_log_file:
        paths.append(os.path.abspath(os.path.expanduser(primary_log_file)))
    seen = set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


def get_data_path(dataset):
    """获取数据集路径"""
    return os.path.join(get_data_dir(), dataset)


def get_merged_path(task_idx):
    """获取合并数据目录中的任务子目录"""
    return os.path.join(get_merged_data_dir(), str(task_idx))


def get_t5_tokenizer_path():
    """T5 tokenizer 本地路径（与 Step5 文本模型目录一致）。"""
    return get_step5_text_model_dir()


def get_mpnet_path():
    """句向量模型本地路径（与 ODCR_SENTENCE_EMBED_MODEL 一致）。"""
    return get_sentence_embed_model_dir()
