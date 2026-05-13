"""MAINLINE 运行前共性校验：路径、参数组合、清晰错误信息。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from odcr_core import path_layout, run_naming
from odcr_core.artifacts import train_csv_path
from paths_config import get_data_dir, get_merged_data_dir

ResolvedConfig = Any

_DOC_MAIN = "docs/ODCR_Scripts_and_Runtime_Guide.md"
_DOC_CONFIG = "configs/odcr.yaml"
_LEGACY_STEP3_KEYS = ("domain_profiles", "user_profiles", "item_profiles")


def _hint_tail() -> str:
    return f"文档: {_DOC_MAIN}；配置: {_DOC_CONFIG}；解析快照见 stdout 或与当次 run 同目录的 manifest.json。"


def _default_data_root_findings() -> list[str]:
    findings: list[str] = []
    checks = (
        ("ODCR_DATA_DIR", Path(get_data_dir()), "Step4 data root"),
        ("ODCR_MERGED_DATA_DIR", Path(get_merged_data_dir()), "Step4 merged root"),
    )
    for env_name, path_obj, label in checks:
        if path_obj.is_dir():
            continue
        raw = (os.environ.get(env_name) or "").strip()
        if raw:
            findings.append(
                f"{label} 缺失：环境变量 {env_name} 当前解析到不存在的目录。\n"
                f"   - {env_name}={raw}\n"
                f"   - 解析后路径: {path_obj}\n"
                f"   - 下一步: 修正 {env_name} 指向当前 ODCR 主线资产根。"
            )
        else:
            findings.append(
                f"{label} 缺失：当前仓库未设置 {env_name}，因此必须存在默认目录。\n"
                f"   - 期望默认路径: {path_obj}\n"
                f"   - 下一步: 补齐该默认目录，或设置 {env_name} 指向当前 ODCR 主线资产根。"
            )
    return findings


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _checkpoint_shape_summary(ckpt_path: Path) -> tuple[list[str] | None, str | None]:
    try:
        import torch
    except ImportError as exc:
        return None, f"无法导入 torch 以检查 checkpoint 形态: {exc}"

    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception as exc:
        return None, f"checkpoint 无法读取: {exc}"

    state_dict: Any = raw
    if isinstance(raw, dict) and isinstance(raw.get("state_dict"), dict):
        state_dict = raw["state_dict"]
    if not isinstance(state_dict, dict):
        return None, f"checkpoint 顶层不是 dict/state_dict（当前类型: {type(state_dict).__name__}）"

    keys = [str(k) for k in state_dict.keys()]
    return keys, None


def _legacy_checkpoint_findings(ckpt_path: Path) -> list[str]:
    findings: list[str] = []
    keys, err = _checkpoint_shape_summary(ckpt_path)
    if err is not None:
        findings.append(
            "Step3 checkpoint 形态检查失败：无法在前置阶段确认它是否为当前主线可消费格式。\n"
            f"   - checkpoint: {ckpt_path}\n"
            f"   - 错误: {err}\n"
            "   - 下一步: 提供当前 ODCR 主线重新产出的 Step3 canonical checkpoint（model/best.pth）。"
        )
        return findings

    assert keys is not None
    legacy_hits = [k for k in _LEGACY_STEP3_KEYS if k in keys]
    if legacy_hits:
        findings.append(
            "检测到旧 3-tensor Step3 checkpoint，当前 Step4 主线禁止继续消费。\n"
            f"   - checkpoint: {ckpt_path}\n"
            f"   - 命中的旧 key: {legacy_hits}\n"
            f"   - 前若干 key: {keys[:12]}\n"
            "   - 原因: 旧 3-tensor profile 与当前 physical_separate / csb_odcr_index_contract/3.0 主线不兼容。\n"
            "   - 下一步: 先生成当前 ODCR 主线的 Step3 checkpoint，再重试 Step4。"
        )
    return findings


def _manifest_lineage_findings(manifest_path: Path, repo_root: Path) -> list[str]:
    findings: list[str] = []
    manifest = _read_json_mapping(manifest_path)
    if manifest is None:
        return findings
    repo_raw = str(manifest.get("repo_root") or "").strip()
    cli_field, cli_raw = _manifest_invocation_info(manifest)
    expected_repo = str(repo_root.resolve())
    repo_mismatch = bool(repo_raw) and Path(repo_raw).expanduser().resolve() != repo_root.resolve()
    old_cli = "code/d4c.py" in cli_raw
    if not (repo_mismatch or old_cli):
        return findings
    findings.append(
        "检测到旧 lineage 的 Step3 上游，Step4 在前置阶段直接拒绝。\n"
        f"   - manifest: {manifest_path}\n"
        f"   - repo_root: {repo_raw or '(缺失)'}\n"
        f"   - 当前 repo_root: {expected_repo}\n"
        f"   - invocation_field: {cli_field}\n"
        f"   - invocation_value: {cli_raw or '(缺失)'}\n"
        "   - 原因: 当前 Step4 只接受由 ODCR-main 产出的 Step3 upstream，不接受旧 D4C-main / d4c.py lineage。"
    )
    return findings


def _manifest_invocation_info(manifest: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("cli_invocation", "invoked_command_line", "invoked_command"):
        raw = str(manifest.get(key) or "").strip()
        if raw:
            return key, raw
    return "(缺失)", ""


def _external_upstream_findings(cfg: ResolvedConfig, step3_dir: Path) -> list[str]:
    findings: list[str] = []
    if not cfg.from_run:
        return findings

    search_root = cfg.repo_root.parent.parent.resolve()
    if not search_root.is_dir():
        return findings

    pattern = f"**/runs/step3/task{cfg.task_id}/{cfg.from_run}/meta/manifest.json"
    manifests = sorted(p.resolve() for p in search_root.glob(pattern))
    local_manifest = (path_layout.logs_dir(step3_dir) / "manifest.json").resolve()
    external = [p for p in manifests if p != local_manifest]
    if not external:
        return findings

    details: list[str] = []
    for manifest_path in external[:3]:
        run_root = manifest_path.parent.parent if manifest_path.parent.name == "meta" else manifest_path.parent
        manifest = _read_json_mapping(manifest_path) or {}
        repo_raw = str(manifest.get("repo_root") or "").strip()
        cli_field, cli_raw = _manifest_invocation_info(manifest)
        ckpt_candidates = [
            run_root / "model" / "best.pth",
            run_root / "model" / "model.pth",
        ]
        ckpt_path = next((p for p in ckpt_candidates if p.is_file()), None)
        detail = [
            f"发现外部 Step3 upstream 候选: {run_root}",
            f"repo_root={repo_raw or '(缺失)'}",
            f"{cli_field}={cli_raw or '(缺失)'}",
        ]
        if ckpt_path is not None:
            detail.append(f"checkpoint={ckpt_path.name}")
            keys, err = _checkpoint_shape_summary(ckpt_path)
            if err is not None:
                detail.append(f"checkpoint_check_error={err}")
            elif keys is not None:
                legacy_hits = [k for k in _LEGACY_STEP3_KEYS if k in keys]
                detail.append(f"legacy_3tensor_keys={legacy_hits or '[]'}")
        else:
            detail.append("checkpoint=(缺失)")
        details.append("   - " + "\n   - ".join(detail))

    findings.append(
        "当前仓库找不到合法的本地 Step3 run，但在工作区上级看到了外部 upstream 候选；这通常意味着你正试图误用旧仓库 lineage。\n"
        f"   - 当前期望本地目录: {step3_dir}\n"
        f"   - 搜索根: {search_root}\n"
        + "\n".join(details)
        + "\n   - 下一步: 先在当前 ODCR-main 下生成合法的 runs/step3/task{T}/<run>/。"
    )
    return findings


def _step4_preflight_findings(cfg: ResolvedConfig) -> list[str]:
    findings = _default_data_root_findings()
    step3_dir = Path(cfg.step3_checkpoint_dir or "")
    step3_model = path_layout.model_file_path(step3_dir)
    manifest_path = path_layout.logs_dir(step3_dir) / "manifest.json"

    if not step3_dir.is_dir():
        findings.append(
            "当前仓库缺少 Step4 所需的本地 Step3 upstream 目录。\n"
            f"   - 期望目录: {step3_dir}\n"
            f"   - task={cfg.task_id} iter={cfg.iteration_id} from_run={cfg.from_run!r}\n"
            "   - 下一步: 先在当前 ODCR-main 下完成对应的 Step3 训练 run。"
        )
        findings.extend(_external_upstream_findings(cfg, step3_dir))
        return findings

    findings.extend(_manifest_lineage_findings(manifest_path, cfg.repo_root))

    if not step3_model.is_file():
        legacy_model = step3_dir / "model" / "model.pth"
        if legacy_model.is_file():
            findings.append(
                "Step3 权重文件名仍为旧布局 model/model.pth，当前主线要求 canonical 路径 model/best.pth。\n"
                f"   - 期望路径: {step3_model}\n"
                f"   - 实际发现: {legacy_model}\n"
                "   - 原因: 这通常表示旧 D4C-main upstream 或旧 checkpoint 产物。\n"
                "   - 下一步: 先生成当前 ODCR 主线 canonical checkpoint，再重试 Step4。"
            )
            findings.extend(_legacy_checkpoint_findings(legacy_model))
        else:
            findings.append(
                "step4 需要 Step3 已产出的 canonical 权重 model/best.pth，但当前目录中缺失。\n"
                f"   - 期望路径: {step3_model}\n"
                "   - 下一步: 先完成当前 ODCR 主线 Step3，确保产出 model/best.pth。"
            )
        return findings

    findings.extend(_legacy_checkpoint_findings(step3_model))
    return findings


def _format_step4_preflight_error(cfg: ResolvedConfig, findings: list[str]) -> str:
    head = (
        "step4 前置校验失败：当前上游/数据资产不满足 ODCR 新主线要求，"
        "已在真正运行前停止，避免旧资产污染继续流入 Step4。\n"
        "已检查:\n"
        "  - 默认 data root(<repo>/data) / merged root(<repo>/merged) 是否存在\n"
        "  - 当前仓库下的 Step3 upstream 路径是否存在\n"
        "  - Step3 manifest 的 repo_root / invocation 字段（cli_invocation、invoked_command_line、invoked_command）lineage\n"
        "  - Step3 checkpoint 是否为 canonical model/best.pth，且不含旧 3-tensor key\n"
        "发现:\n"
    )
    body = "\n".join(f"  {idx}. {item}" for idx, item in enumerate(findings, start=1))
    tail = (
        "\n下一步建议:\n"
        f"  - 为 task={cfg.task_id} iter={cfg.iteration_id} 在当前 ODCR-main 下生成合法的 Step3 upstream\n"
        "  - 补齐 <repo>/data 与 <repo>/merged，或显式设置 ODCR_DATA_DIR / ODCR_MERGED_DATA_DIR 指向当前主线资产\n"
        "  - 不要混用旧 D4C-main lineage、旧 model/model.pth、旧 3-tensor checkpoint\n"
        + _hint_tail()
    )
    return head + body + tail


def validate_resolved_config(cfg: ResolvedConfig) -> None:
    """
    在 config_resolver.resolve_config 之后、torchrun 之前调用。
    不替代 YAML/任务表解析错误（仍由 config_resolver 抛出）。
    """
    cmd = cfg.command
    if cmd in ("step3", "step5", "eval", "eval-rerank"):
        if not (getattr(cfg, "effective_training_payload_json", "") or "").strip():
            raise RuntimeError(
                f"内部错误: command={cmd!r} 缺少 effective_training_payload_json（父进程须生成训练 payload）。"
            )
    ck = Path(cfg.checkpoint_dir)
    it = cfg.iteration_id
    t = cfg.task_id

    if cmd == "step4":
        upstream_resolution = (getattr(cfg, "upstream_resolution_json", "") or "").strip()
        if not upstream_resolution:
            raise RuntimeError("内部错误: step4 缺少 upstream_resolution_json；Step4 dry-run/runtime 必须复用 upstream_resolver。")
        s3 = Path(cfg.step3_checkpoint_dir or "")
        preflight_findings = _step4_preflight_findings(cfg)
        if preflight_findings:
            raise RuntimeError(_format_step4_preflight_error(cfg, preflight_findings))
        _eid = (getattr(cfg, "eval_profile_id", "") or "").strip()
        if not _eid:
            raise RuntimeError(
                "内部错误: step4 缺少 eval_profile_id（须由 CLI --eval-profile 解析；step4 已归入 eval 语义侧）。"
            )
        if cfg.global_eval_batch_size is None:
            raise RuntimeError(
                "内部错误: step4 缺少 global_eval_batch_size（须由 eval_profile.eval_batch_size 解析）。"
            )
        if int(cfg.global_eval_batch_size) % int(cfg.ddp_world_size) != 0:
            raise ValueError(
                f"step4: eval_profile「{_eid}」中的 eval_batch_size={int(cfg.global_eval_batch_size)} "
                f"不能整除当前 hardware 预设的 ddp_world_size={int(cfg.ddp_world_size)}。\n"
                f"请修改 configs/odcr.yaml 中 eval.profiles.{_eid}.eval_batch_size，"
                f"或修改 hardware.profiles.{cfg.hardware_preset_id}.ddp_world_size。\n"
                + _hint_tail()
            )
    elif cmd == "step5":
        upstream_resolution = (getattr(cfg, "upstream_resolution_json", "") or "").strip()
        if not upstream_resolution:
            raise RuntimeError("内部错误: step5 缺少 upstream_resolution_json；Step5 必须复用 upstream_resolver。")
        assert cfg.from_run is not None
        rid = run_naming.parse_run_id(cfg.from_run)
        step3_dir = path_layout.get_train_step3_run_root(cfg.repo_root, cfg.task_id, cfg.iteration_id, rid)
        if not step3_dir.is_dir():
            raise FileNotFoundError(
                f"step5 需要已存在的 Step3 目录:\n  {step3_dir}\n"
                f"task={t} iter={it} from_run={cfg.from_run!r}\n"
                "请检查 --from-run。\n"
                + _hint_tail()
            )
        assert cfg.step5_run is not None
        s4_slug = run_naming.step4_slug_from_step5_slug(cfg.step5_run)
        step4_dir = path_layout.get_train_step4_run_root(
            cfg.repo_root, cfg.task_id, cfg.iteration_id, s4_slug
        )
        csv_p = train_csv_path(cfg)
        if not csv_p.is_file():
            raise FileNotFoundError(
                "step5 需要 Step4 正式训练表 odcr_routing_train.csv（由 step5_run 经 run_naming 反推 step4）：\n"
                f"  期望文件: {csv_p}\n"
                f"  task={t} iter={it} step5_run={cfg.step5_run!r} → step4_run={s4_slug!r}\n"
                f"  对应目录应为: {step4_dir}\n"
                "请先完成 step4，或使 Step5 目录名与 Step4 一致（形如 {{step4}}_{{n}}，例如 step4=2_1 → step5=2_1_1）。\n"
                + _hint_tail()
            )
    elif cmd == "step3":
        if cfg.step3_mode == "eval_only":
            mp = path_layout.model_file_path(ck)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"step3 --eval-only 需要已有权重:\n  {mp}\n"
                    "请先跑完整 step3 训练，或去掉 --eval-only。\n"
                    f"评测日志: {cfg.log_dir}/full.log\n"
                    + _hint_tail()
                )
    elif cmd in ("eval", "eval-rerank"):
        if cfg.global_eval_batch_size is None:
            raise RuntimeError("内部错误: eval 系命令缺少 global_eval_batch_size。")
        if int(cfg.global_eval_batch_size) % int(cfg.ddp_world_size) != 0:
            raise ValueError(
                f"eval_batch_size={int(cfg.global_eval_batch_size)} 与 world_size={int(cfg.ddp_world_size)} 不整除。"
                "请修改 configs/odcr.yaml 中 eval.profiles.*.eval_batch_size，或调整 hardware.profiles.*.ddp_world_size。"
            )
        stage = "rerank" if cmd == "eval-rerank" else "eval"
        if cfg.model_path:
            mp = Path(cfg.model_path)
            if not mp.is_file():
                raise FileNotFoundError(
                    f"eval --model-path 不是有效文件:\n  {mp}\n"
                    "请核对路径或改用 --from-run + --step5-run。\n"
                    + _hint_tail()
                )
        else:
            mp = path_layout.model_file_path(ck)
            if not mp.is_file():
                er = cfg.eval_run_dir or "(尚未分配)"
                raise FileNotFoundError(
                    f"{cmd} 需要 Step5 训练权重（默认 model/best.pth）：\n"
                    f"  {mp}\n"
                    f"  task={t} iter={it} from_run={cfg.from_run!r} step5_run={cfg.step5_run!r}\n"
                    f"  本次将写入: runs/{stage}/task{t}/<run>/（当前 eval_run_dir={er}）\n"
                    "请确认已完成 step5，且 --from-run / --step5-run 与训练时一致。\n"
                    + _hint_tail()
                )
