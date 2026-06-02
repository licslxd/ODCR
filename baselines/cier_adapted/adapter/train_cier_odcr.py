from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    BASELINE,
    decode_profile,
    ensure_run_layout,
    load_config,
    normalize_mode,
    read_json,
    read_jsonl,
    run_dir,
    sha256_file,
    task_domains,
    utc_now,
    write_json,
    write_jsonl,
    write_resolved_config,
)


def _records_path(run_path: Path, name: str) -> Path:
    return run_path / "data" / "cier_records" / f"{name}.jsonl"


def _load_records(run_path: Path, name: str) -> list[dict[str, Any]]:
    path = _records_path(run_path, name)
    if not path.is_file():
        raise FileNotFoundError(f"CIER dataset is missing: {path}. Run build_cier_dataset.py first.")
    return read_jsonl(path)


class CIERRecordDataset:
    def __init__(self, path: Path, *, limit: int | None = None) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        if limit is not None and int(limit) <= 0:
            self._fh = None
            return
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
                    if limit is not None and len(self.offsets) >= int(limit):
                        break
        self._fh = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_fh"] = None
        return state

    def _handle(self):
        if self._fh is None:
            self._fh = self.path.open("rb")
        return self._fh

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fh = self._handle()
        fh.seek(self.offsets[int(idx)])
        return json.loads(fh.readline().decode("utf-8"))


class CIERBatchCollater:
    def __init__(
        self,
        *,
        tokenizer: Any,
        max_step: int = 1,
        word: int = 20,
        delta: float = 0.5,
        include_meta: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_step = max(1, int(max_step))
        self.cur_step = 1
        self.word = int(word)
        self.delta = float(delta)
        self.include_meta = include_meta
        self.eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    def _ids(self, text: str) -> list[int]:
        ids = self.tokenizer(str(text))["input_ids"][1 : self.word]
        if not ids:
            ids = [self.eos]
        elif ids[-1] != self.eos:
            ids = list(ids) + [self.eos]
        return [int(x) for x in ids[: self.word]]

    @staticmethod
    def _rating_input(rating: int, delta: float) -> list[float]:
        import numpy as np

        if np.random.rand() < delta and 0 < rating < 4:
            values = [0.0, 0.0, 0.0, 0.0, 0.0]
            rand = np.random.rand() * 2 / 3
            values[int(rating)] = 1 - rand
            values[int(rating) - 1] = rand / 2
            values[int(rating) + 1] = rand / 2
            return values
        values = [0.0, 0.0, 0.0, 0.0, 0.0]
        values[max(0, min(4, int(rating)))] = 1.0
        return values

    def __call__(self, data: list[dict[str, Any]]):
        import numpy as np
        import torch

        tokenized = []
        for row in data:
            tokenized.append(
                {
                    "text": self._ids(str(row["explanation"])),
                    "keyword": self._ids(str(row["keyword"])),
                    "user": int(row["cier_user"]),
                    "item": int(row["cier_item"]),
                    "rating": int(row["rating_index"]),
                    "_meta": row,
                }
            )
        max_length = max(min(self.word, len(x["text"])) for x in tokenized)
        input_ids, userid, itemid, curr_flag, rating, rating_inputs = [], [], [], [], [], []
        metas = []
        for row in tokenized:
            if np.random.rand() < self.cur_step / self.max_step:
                ids = row["text"][:max_length]
                curr_flag.append(1)
            else:
                ids = row["keyword"][:max_length]
                curr_flag.append(0)
            ids = ids + [0] * (max_length - len(ids))
            input_ids.append(ids)
            userid.append(row["user"])
            itemid.append(row["item"])
            rating.append(row["rating"])
            rating_inputs.append(self._rating_input(row["rating"], self.delta))
            metas.append(row["_meta"])
        self.cur_step += 1
        batch = (
            torch.tensor(input_ids),
            torch.tensor(userid),
            torch.tensor(itemid),
            torch.tensor(rating).long(),
            torch.tensor(curr_flag),
            torch.tensor(rating_inputs),
        )
        if self.include_meta:
            return batch + (metas,)
        return batch


def _training_plan(config: dict[str, Any], mode: str, run_path: Path) -> dict[str, Any]:
    training = dict(config.get("training") or {})
    source_domain, target_domain = task_domains(config)
    phases = []
    if mode == "source_to_target":
        phases.append(
            {
                "phase": "source_pretrain",
                "domain": source_domain,
                "input": str(_records_path(run_path, "source_train")),
                "epochs": int(training.get("source_epochs", 1)),
            }
        )
    phases.append(
        {
            "phase": "target_finetune",
            "domain": target_domain,
            "input": str(_records_path(run_path, "target_train")),
            "valid": str(_records_path(run_path, "target_valid")),
            "epochs": int(training.get("target_epochs", training.get("epochs", 3))),
        }
    )
    return {
        "schema_version": "odcr_cier_training_plan_v1",
        "baseline": BASELINE,
        "mode": mode,
        "model_name": training.get("model_name", "openlm-research/open_llama_7b_v2"),
        "batch_size": int(training.get("batch_size", 64)),
        "eval_batch_size": int(training.get("eval_batch_size", training.get("batch_size", 64))),
        "learning_rate": float(training.get("learning_rate", 1.0e-3)),
        "accumulation_steps": int(training.get("accumulation_steps", 1)),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": bool(training.get("pin_memory", True)),
        "persistent_workers": bool(training.get("persistent_workers", False)),
        "prefetch_factor": training.get("prefetch_factor", None),
        "precision": str(training.get("precision", "fp16")),
        "tf32": bool(training.get("tf32", True)),
        "device_index": int(training.get("device_index", 0)),
        "word": int(decode_profile(config)["max_length"]),
        "target_checkpoint_selection": "lowest_target_valid_loss",
        "phases": phases,
        "full_training_requires_cuda": True,
        "odcr_active_path_modified": False,
    }


def _raw_prediction_from_record(row: dict[str, Any], *, split: str, source_domain: str, target_domain: str) -> dict[str, Any]:
    return {
        "schema_version": "cier_adapted_raw_prediction_v1",
        "baseline": BASELINE,
        "task_id": int(row["task_id"]),
        "split": split,
        "user_id": str(row["user_id"]),
        "item_id": str(row["item_id"]),
        "source_domain": source_domain,
        "target_domain": target_domain,
        "rating_gold": float(row["rating"]),
        "rating_pred": float(row["rating"]),
        "reference": str(row["explanation"]),
        "prediction": str(row["explanation"]),
        "raw_prediction_source": "smoke_reference_copy",
    }


def _run_smoke(args: argparse.Namespace, config: dict[str, Any], mode: str, run_path: Path) -> dict[str, Any]:
    source_domain, target_domain = task_domains(config)
    source_train = _load_records(run_path, "source_train") if mode == "source_to_target" else []
    target_train = _load_records(run_path, "target_train")
    target_valid = _load_records(run_path, "target_valid")
    target_test = _load_records(run_path, "target_test")
    max_steps = int(args.max_steps or 1)
    model_payload = {
        "schema_version": "odcr_cier_smoke_checkpoint_v1",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": args.run_id,
        "mode": mode,
        "source_train_rows_seen": min(len(source_train), max_steps) if source_train else 0,
        "target_train_rows_seen": min(len(target_train), max_steps),
        "target_valid_rows_available": len(target_valid),
        "target_test_rows_available": len(target_test),
        "real_cier_training": False,
        "full_training_started": False,
        "created_at": utc_now(),
    }
    checkpoint_path = run_path / "model" / "smoke_checkpoint.json"
    write_json(checkpoint_path, model_payload)
    for split, records in (("valid", target_valid), ("test", target_test)):
        raw_rows = [
            _raw_prediction_from_record(row, split=split, source_domain=source_domain, target_domain=target_domain)
            for row in records
        ]
        write_jsonl(run_path / "model" / f"cier_raw_{split}_predictions.jsonl", raw_rows)
    status = {
        "schema_version": "odcr_cier_stage_status_v1",
        "status": "train_smoke_completed",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": args.run_id,
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "selected_checkpoint": str(checkpoint_path),
        "selection_split": "target_valid",
        "real_cier_training": False,
        "full_training_started": False,
        "updated_at": utc_now(),
    }
    write_json(run_path / "meta" / "stage_status.json", status)
    _update_run_summary(run_path, status=status)
    print(json.dumps({"status": status["status"], "checkpoint": str(checkpoint_path)}, indent=2))
    return status


def _update_run_summary(run_path: Path, *, status: dict[str, Any]) -> None:
    summary_path = run_path / "meta" / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
    else:
        summary = {"schema_version": "odcr_cier_run_summary_v1", "baseline": BASELINE}
    summary.update(
        {
            "status": status.get("status"),
            "stage_status": "meta/stage_status.json",
            "model": {
                "checkpoint": status.get("checkpoint"),
                "selected_checkpoint": status.get("selected_checkpoint"),
                "real_cier_training": bool(status.get("real_cier_training", False)),
            },
            "updated_at": utc_now(),
        }
    )
    write_json(summary_path, summary)


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "Current tmux does not expose CUDA. Please manually run `odcr-enter-gpu <JOBID>` "
            "in this same tmux to enter the GPU node, then rerun the probe."
        )


def _tokenized_frame(records: list[dict[str, Any]], tokenizer: Any, *, max_len: int):
    import pandas as pd

    eos = tokenizer.eos_token_id
    if eos is None:
        eos = 2
    rows: list[dict[str, Any]] = []
    for row in records:
        text_ids = tokenizer(str(row["explanation"]))["input_ids"][1:max_len]
        keyword_ids = tokenizer(str(row["keyword"]))["input_ids"][1:max_len]
        rows.append(
            {
                "user": int(row["cier_user"]),
                "item": int(row["cier_item"]),
                "text": [int(x) for x in text_ids] + [int(eos)],
                "keyword": [int(x) for x in keyword_ids] or [int(eos)],
                "keyword_words": str(row["keyword_words"]),
                "rating": int(row["rating_index"]),
                "_meta": row,
            }
        )
    return pd.DataFrame(rows)


def _make_loader(
    df: Any,
    collater: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
):
    from torch.utils.data import DataLoader

    upstream_dataset = sys.modules["dataloader"].MyDataset(df.drop(columns=["_meta"]))
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collater,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(upstream_dataset, **kwargs)


def _make_record_loader(
    dataset: CIERRecordDataset,
    collater: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
):
    from torch.utils.data import DataLoader

    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "collate_fn": collater,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _predict_full(model: Any, dataset: CIERRecordDataset, tokenizer: Any, config: dict[str, Any], *, split: str) -> list[dict[str, Any]]:
    import numpy as np
    import torch
    from utils import ids2words, ids_clear

    profile = decode_profile(config)
    word = int(profile["max_length"])
    training = dict(config.get("training") or {})
    precision = str(training.get("precision", "fp16")).lower()
    autocast_dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32
    autocast_enabled = precision in {"fp16", "bf16"}
    batch_size = int(training.get("eval_batch_size", training.get("batch_size", 64)))
    loader = _make_record_loader(
        dataset,
        CIERBatchCollater(tokenizer=tokenizer, max_step=1, word=word, include_meta=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(training.get("eval_num_workers", training.get("num_workers", 0))),
        pin_memory=bool(training.get("pin_memory", True)),
        persistent_workers=bool(training.get("persistent_workers", False)),
        prefetch_factor=training.get("prefetch_factor", None),
    )
    source_domain, target_domain = task_domains(config)
    model.eval()
    out: list[dict[str, Any]] = []
    device = next(model.parameters()).device
    for batch in loader:
        input_ids, userid, itemid, rating, curr_flag, rating_inputs, metas = batch
        del input_ids, curr_flag, rating_inputs
        userid = userid.to(device)
        itemid = itemid.to(device)
        rating = rating.to(device)
        text = torch.tensor([[]], dtype=torch.long, device=device)
        last_words = torch.tensor([[]], dtype=torch.long, device=device)
        kv_cache = None
        rating_pred_values: list[float] = []
        for idx in range(word):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                    if idx == 0:
                        pre_rating = model.rating_predict(userid, itemid)
                        batch_pred = pre_rating.float().detach().cpu().numpy()
                        rating_pred_values = [
                            (item * np.array([1.0, 2.0, 3.0, 4.0, 5.0])).sum().item() for item in batch_pred
                        ]
                    logits, kv_cache = model(last_words, userid, itemid, pre_rating, kv_cache)
            last_words = torch.argmax(logits, dim=1).unsqueeze(1)
            text = last_words if text.shape[1] == 0 else torch.cat([text, last_words], 1)
        decoded = [" ".join(ids2words(ids_clear(ids), tokenizer)) for ids in text.detach().cpu().tolist()]
        for meta, pred_text, pred_rating in zip(metas, decoded, rating_pred_values):
            out.append(
                {
                    "schema_version": "cier_adapted_raw_prediction_v1",
                    "baseline": BASELINE,
                    "task_id": int(meta["task_id"]),
                    "split": split,
                    "user_id": str(meta["user_id"]),
                    "item_id": str(meta["item_id"]),
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "rating_gold": float(meta["rating"]),
                    "rating_pred": float(pred_rating),
                    "reference": str(meta["explanation"]),
                    "prediction": pred_text,
                    "raw_prediction_source": "cier_full_generation",
                }
            )
    return out


def _apply_training_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    resolved = dict(config)
    training = dict(resolved.get("training") or {})
    override_map = {
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "accumulation_steps": args.accumulation_steps,
        "num_workers": args.num_workers,
        "eval_num_workers": args.eval_num_workers,
        "prefetch_factor": args.prefetch_factor,
        "precision": args.precision,
        "device_index": args.device_index,
        "source_epochs": args.source_epochs,
        "target_epochs": args.target_epochs,
        "show_train_loss_steps": args.show_train_loss_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    for key, value in override_map.items():
        if value is not None:
            training[key] = value
    if args.pin_memory is not None:
        training["pin_memory"] = bool(args.pin_memory)
    if args.persistent_workers is not None:
        training["persistent_workers"] = bool(args.persistent_workers)
    if args.tf32 is not None:
        training["tf32"] = bool(args.tf32)
    resolved["training"] = training
    return resolved


def _autocast_context(torch_module: Any, *, precision: str, device_type: str = "cuda"):
    if precision == "fp32":
        return torch_module.autocast(device_type=device_type, enabled=False)
    dtype = torch_module.bfloat16 if precision == "bf16" else torch_module.float16
    return torch_module.autocast(device_type=device_type, dtype=dtype)


def _train_loop(
    model: Any,
    train_dataloader: Any,
    optimizer: Any,
    *,
    device: Any,
    epoch: int,
    show_train_loss_steps: int,
    accumulation_steps: int,
    scaler: Any,
    log_name: str,
    precision: str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    import torch
    from tqdm import tqdm

    model.train()
    loss_log: list[float] = []
    total_loss = 0.0
    total_samples = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    completed_steps = 0
    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        input_ids, userid, itemid, rating, curr_flag, rating_inputs = batch
        input_ids = input_ids.to(device, non_blocking=True)
        itemid = itemid.to(device, non_blocking=True)
        userid = userid.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True)
        curr_flag = curr_flag.to(device, non_blocking=True)
        rating_inputs = rating_inputs.to(device, non_blocking=True)
        with _autocast_context(torch, precision=precision):
            loss = model.train_step(input_ids, userid, itemid, rating, curr_flag, rating_inputs)
        loss_value = float(loss.detach().cpu().item())
        loss_log.append(loss_value)
        total_loss += loss_value
        total_samples += int(input_ids.shape[0])
        scaled_loss = loss / max(int(accumulation_steps), 1)
        if precision == "fp16":
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (batch_idx + 1) % max(int(accumulation_steps), 1) == 0
        if should_step:
            if precision == "fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if precision == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        completed_steps = batch_idx + 1
        if completed_steps % max(int(show_train_loss_steps), 1) == 0:
            message = (
                "Train Epoch: {} [{}/{} ({}%)]\t Loss: {}\n".format(
                    epoch,
                    completed_steps * input_ids.shape[0],
                    len(train_dataloader.dataset),
                    round(100.0 * batch_idx / max(len(train_dataloader), 1), 2),
                    round(sum(loss_log) / max(len(loss_log), 1), 6),
                )
            )
            with open(log_name, "a+", encoding="utf-8") as fh:
                fh.write(message)
            print(message.strip())
            loss_log = []
        if max_steps is not None and completed_steps >= int(max_steps):
            break

    if completed_steps and completed_steps % max(int(accumulation_steps), 1) != 0:
        if precision == "fp16":
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    elapsed = max(time.time() - started, 1e-9)
    return {
        "steps": completed_steps,
        "samples": total_samples,
        "seconds": elapsed,
        "samples_per_second": total_samples / elapsed,
        "steps_per_second": completed_steps / elapsed,
        "avg_loss": total_loss / max(completed_steps, 1),
    }


def _valid_loop(model: Any, valid_dataloader: Any, *, device: Any, log_name: str, precision: str, max_steps: int | None = None) -> float:
    import torch

    model.eval()
    loss_log: list[float] = []
    for batch_idx, batch in enumerate(valid_dataloader):
        input_ids, userid, itemid, rating, curr_flag, rating_inputs = batch
        input_ids = input_ids.to(device, non_blocking=True)
        itemid = itemid.to(device, non_blocking=True)
        userid = userid.to(device, non_blocking=True)
        rating = rating.to(device, non_blocking=True)
        curr_flag = curr_flag.to(device, non_blocking=True)
        rating_inputs = rating_inputs.to(device, non_blocking=True)
        with torch.no_grad():
            with _autocast_context(torch, precision=precision):
                loss = model.train_step(input_ids, userid, itemid, rating, curr_flag, rating_inputs)
        loss_log.append(float(loss.detach().cpu().item()))
        if max_steps is not None and batch_idx + 1 >= int(max_steps):
            break
    value = round(sum(loss_log) / max(len(loss_log), 1), 6)
    with open(log_name, "a+", encoding="utf-8") as fh:
        fh.write(f"valid Loss: {value}\n")
    print(f"valid Loss: {value}")
    return value


def _loader_options(training: dict[str, Any], *, eval_loader: bool = False) -> dict[str, Any]:
    workers_key = "eval_num_workers" if eval_loader else "num_workers"
    return {
        "num_workers": int(training.get(workers_key, training.get("num_workers", 0))),
        "pin_memory": bool(training.get("pin_memory", True)),
        "persistent_workers": bool(training.get("persistent_workers", False)),
        "prefetch_factor": training.get("prefetch_factor", None),
    }


def _maybe_subset(records: list[dict[str, Any]], *, enabled: bool, rows: int) -> list[dict[str, Any]]:
    if not enabled:
        return records
    return records[: max(1, int(rows))]


def _id_space_from_source_table(run_path: Path) -> tuple[int, int] | None:
    source_table_path = run_path / "meta" / "source_table.json"
    if not source_table_path.is_file():
        return None
    source_table = read_json(source_table_path)
    active = dict(source_table.get("active_id_space") or {})
    if "user_count" in active and "item_count" in active:
        return int(active["user_count"]), int(active["item_count"])
    return None


def _scan_id_space(paths: list[Path]) -> tuple[int, int]:
    max_user = -1
    max_item = -1
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                max_user = max(max_user, int(row["cier_user"]))
                max_item = max(max_item, int(row["cier_item"]))
    return max_user + 1, max_item + 1


def _valid_losses_from_log(log_path: Path) -> list[float]:
    losses: list[float] = []
    if not log_path.is_file():
        return losses
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("valid Loss:"):
            continue
        try:
            losses.append(float(line.split(":", 1)[1].strip()))
        except ValueError:
            continue
    return losses


def _run_predict_only(
    args: argparse.Namespace,
    config: dict[str, Any],
    mode: str,
    run_path: Path,
    data_run_path: Path,
) -> dict[str, Any]:
    _require_cuda()
    import torch
    from peft import PeftModel
    from transformers import LlamaForCausalLM, LlamaTokenizer

    upstream = Path(config["upstream"]["path"])
    if not upstream.is_absolute():
        upstream = Path(__file__).resolve().parents[3] / upstream
    sys.path.insert(0, str(upstream))
    from model import MyModel  # noqa: WPS433

    training = dict(config.get("training") or {})
    model_name = str(training.get("model_name", "openlm-research/open_llama_7b_v2"))
    precision = str(training.get("precision", "fp16")).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision {precision!r}; expected fp16, bf16, or fp32")
    device_index = int(training.get("device_index", 0))
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"device_index {device_index} outside visible CUDA device count {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    if bool(training.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    checkpoint_dir = run_path / "model" / "best_lora"
    prompt_path = run_path / "model" / "best_prompt_encoder.pt"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing CIER LoRA checkpoint: {checkpoint_dir}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Missing CIER prompt encoder checkpoint: {prompt_path}")

    target_valid_path = _records_path(data_run_path, "target_valid")
    target_test_path = _records_path(data_run_path, "target_test")
    active_paths = [_records_path(data_run_path, "target_train"), target_valid_path, target_test_path]
    if mode == "source_to_target":
        active_paths.insert(0, _records_path(data_run_path, "source_train"))
    user_num, item_num = _id_space_from_source_table(data_run_path) or _scan_id_space(active_paths)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32
    base_llm = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": device_index},
    )
    model_llm = PeftModel.from_pretrained(base_llm, str(checkpoint_dir), is_trainable=False)
    model = MyModel(user_num, item_num, int(training.get("id_hidden", 1024)), model_llm.config.hidden_size, tokenizer).to(device)
    model.model = model_llm
    model.generate_weight = float(training.get("generate_weight", 1.0))
    model.rating_weight = float(training.get("rating_weight", 0.1))
    state = torch.load(str(prompt_path), map_location=device)
    model.prompt_encoder.load_state_dict(state)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()

    outputs: dict[str, Any] = {}
    for split, dataset in (
        ("valid", CIERRecordDataset(target_valid_path)),
        ("test", CIERRecordDataset(target_test_path)),
    ):
        raw_rows = _predict_full(model, dataset, tokenizer, config, split=split)
        output_path = run_path / "model" / f"cier_raw_{split}_predictions.jsonl"
        count = write_jsonl(output_path, raw_rows)
        outputs[split] = {"path": str(output_path), "row_count": count}

    valid_losses = _valid_losses_from_log(run_path / "meta" / "cier_train.log")
    status = {
        "schema_version": "odcr_cier_stage_status_v1",
        "status": "trained_and_raw_predictions_exported",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": args.run_id,
        "data_run_id": args.data_run_id or args.run_id,
        "mode": mode,
        "checkpoint": str(checkpoint_dir),
        "selected_checkpoint": str(checkpoint_dir),
        "prompt_encoder": str(prompt_path),
        "checkpoint_sha256": sha256_file(prompt_path),
        "selection_split": "target_valid",
        "best_valid_loss": min(valid_losses) if valid_losses else None,
        "valid_losses": valid_losses,
        "raw_predictions": outputs,
        "training_parameters": {
            "model_name": model_name,
            "batch_size": int(training.get("batch_size", 64)),
            "eval_batch_size": int(training.get("eval_batch_size", training.get("batch_size", 64))),
            "accumulation_steps": int(training.get("accumulation_steps", 1)),
            "global_batch": int(training.get("batch_size", 64)) * int(training.get("accumulation_steps", 1)),
            "num_workers": int(training.get("num_workers", 0)),
            "eval_num_workers": int(training.get("eval_num_workers", training.get("num_workers", 0))),
            "pin_memory": bool(training.get("pin_memory", True)),
            "persistent_workers": bool(training.get("persistent_workers", False)),
            "prefetch_factor": training.get("prefetch_factor", None),
            "precision": precision,
            "tf32": bool(training.get("tf32", True)),
            "device_index": device_index,
            "max_generation_length": int(decode_profile(config)["max_length"]),
        },
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 4),
        "seconds": round(time.time() - started, 4),
        "real_cier_training": True,
        "full_training_started": True,
        "prediction_resume_after_training_failure": True,
        "uses_step3_step4_evidence_routing": False,
        "updated_at": utc_now(),
    }
    write_json(run_path / "meta" / "stage_status.json", status)
    _update_run_summary(run_path, status=status)
    print(json.dumps(status, indent=2, sort_keys=True))
    return status


def _run_full(args: argparse.Namespace, config: dict[str, Any], mode: str, run_path: Path, data_run_path: Path) -> dict[str, Any]:
    _require_cuda()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.cuda.amp import GradScaler
    from torch.optim import AdamW
    from transformers import LlamaForCausalLM, LlamaTokenizer

    upstream = Path(config["upstream"]["path"])
    if not upstream.is_absolute():
        upstream = Path(__file__).resolve().parents[3] / upstream
    sys.path.insert(0, str(upstream))
    from dataloader import MyCollater  # noqa: WPS433
    from model import MyModel  # noqa: WPS433

    training = dict(config.get("training") or {})
    profile = decode_profile(config)
    word = int(profile["max_length"])
    model_name = str(training.get("model_name", "openlm-research/open_llama_7b_v2"))
    batch_size = int(training.get("batch_size", 64))
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    learning_rate = float(training.get("learning_rate", 1.0e-3))
    accumulation_steps = int(training.get("accumulation_steps", 1))
    precision = str(training.get("precision", "fp16")).lower()
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision {precision!r}; expected fp16, bf16, or fp32")
    device_index = int(training.get("device_index", 0))
    if device_index >= torch.cuda.device_count():
        raise ValueError(f"device_index {device_index} outside visible CUDA device count {torch.cuda.device_count()}")
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    if bool(training.get("tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    source_path = _records_path(data_run_path, "source_train")
    target_train_path = _records_path(data_run_path, "target_train")
    target_valid_path = _records_path(data_run_path, "target_valid")
    target_test_path = _records_path(data_run_path, "target_test")
    if args.probe:
        probe_rows = max(batch_size * max(int(args.max_steps or 1), 1) * max(accumulation_steps, 1), batch_size)
        source_limit = probe_rows
        target_train_limit = probe_rows
        target_valid_limit = max(eval_batch_size * 2, batch_size)
        target_test_limit = 0
    else:
        source_limit = None
        target_train_limit = None
        target_valid_limit = None
        target_test_limit = None
    active_paths = [target_train_path, target_valid_path, target_test_path]
    if mode == "source_to_target":
        active_paths.insert(0, source_path)
    id_space = _id_space_from_source_table(data_run_path) or _scan_id_space(active_paths)
    user_num, item_num = id_space

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    source_dataset = (
        CIERRecordDataset(source_path, limit=source_limit) if mode == "source_to_target" and source_path.is_file() else None
    )
    target_train_dataset = CIERRecordDataset(target_train_path, limit=target_train_limit)
    target_valid_dataset = CIERRecordDataset(target_valid_path, limit=target_valid_limit)
    target_test_dataset = CIERRecordDataset(target_test_path, limit=target_test_limit)
    lora_config = LoraConfig(
        r=int(training.get("lora_r", 4)),
        lora_alpha=int(training.get("lora_alpha", 32)),
        target_modules=list(training.get("target_modules", ["q_proj", "k_proj"])),
        lora_dropout=float(training.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    torch_dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32
    model_llm = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map={"": device_index},
    )
    model_llm = get_peft_model(model_llm, lora_config)
    model = MyModel(user_num, item_num, int(training.get("id_hidden", 1024)), model_llm.config.hidden_size, tokenizer).to(device)
    model.model = model_llm
    model.generate_weight = float(training.get("generate_weight", 1.0))
    model.rating_weight = float(training.get("rating_weight", 0.1))
    optimizer = AdamW(
        [
            {"params": filter(lambda p: p.requires_grad, model.prompt_encoder.parameters()), "lr": learning_rate},
            {"params": filter(lambda p: p.requires_grad, model.model.parameters()), "lr": learning_rate / 10.0},
        ]
    )
    scaler = GradScaler(enabled=(precision == "fp16"))
    log_name = str(run_path / "meta" / "cier_train.log")
    show_steps = int(training.get("show_train_loss_steps", 500))
    source_epochs = int(training.get("source_epochs", 1))
    target_epochs = int(training.get("target_epochs", training.get("epochs", 3)))
    max_steps = int(args.max_steps) if (args.probe and args.max_steps) else None
    train_stats: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    run_started = time.time()
    if mode == "source_to_target" and source_dataset is not None and len(source_dataset) > 0:
        collate_source = CIERBatchCollater(
            tokenizer=tokenizer,
            max_step=max(1, source_epochs * len(source_dataset) // max(batch_size, 1)),
            word=word,
            delta=float(training.get("delta", 0.2)),
        )
        source_loader = _make_record_loader(
            source_dataset,
            collate_source,
            batch_size=batch_size,
            shuffle=True,
            **_loader_options(training),
        )
        for epoch in range(source_epochs):
            stats = _train_loop(
                model,
                source_loader,
                optimizer,
                device=device,
                epoch=epoch,
                show_train_loss_steps=show_steps,
                accumulation_steps=accumulation_steps,
                scaler=scaler,
                log_name=log_name,
                precision=precision,
                max_steps=max_steps,
            )
            stats.update({"phase": "source_pretrain", "epoch": epoch})
            train_stats.append(stats)

    collate_train = CIERBatchCollater(
        tokenizer=tokenizer,
        max_step=max(1, target_epochs * len(target_train_dataset) // max(batch_size, 1)),
        word=word,
        delta=float(training.get("delta", 0.2)),
    )
    collate_valid = CIERBatchCollater(tokenizer=tokenizer, max_step=1, word=word)
    target_loader = _make_record_loader(
        target_train_dataset,
        collate_train,
        batch_size=batch_size,
        shuffle=True,
        **_loader_options(training),
    )
    valid_loader = _make_record_loader(
        target_valid_dataset,
        collate_valid,
        batch_size=eval_batch_size,
        shuffle=False,
        **_loader_options(training, eval_loader=True),
    )
    best_loss = float("inf")
    checkpoint_dir = run_path / "model" / "best_lora"
    prompt_path = run_path / "model" / "best_prompt_encoder.pt"
    best_adapter_loaded = False
    for epoch in range(target_epochs):
        stats = _train_loop(
            model,
            target_loader,
            optimizer,
            device=device,
            epoch=epoch,
            show_train_loss_steps=show_steps,
            accumulation_steps=accumulation_steps,
            scaler=scaler,
            log_name=log_name,
            precision=precision,
            max_steps=max_steps,
        )
        stats.update({"phase": "target_finetune", "epoch": epoch})
        train_stats.append(stats)
        valid_loss = _valid_loop(
            model,
            valid_loader,
            device=device,
            log_name=log_name,
            precision=precision,
            max_steps=max_steps if args.probe else None,
        )
        if float(valid_loss) <= best_loss:
            best_loss = float(valid_loss)
            model.model.save_pretrained(str(checkpoint_dir))
            torch.save(model.prompt_encoder.state_dict(), str(prompt_path))

    if args.probe:
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        status = {
            "schema_version": "odcr_cier_probe_status_v1",
            "status": "probe_completed",
            "baseline": BASELINE,
            "task_id": int(args.task),
            "run_id": args.run_id,
            "data_run_id": args.data_run_id or args.run_id,
            "mode": mode,
            "model_name": model_name,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "accumulation_steps": accumulation_steps,
            "global_batch": batch_size * accumulation_steps,
            "num_workers": int(training.get("num_workers", 0)),
            "eval_num_workers": int(training.get("eval_num_workers", training.get("num_workers", 0))),
            "pin_memory": bool(training.get("pin_memory", True)),
            "persistent_workers": bool(training.get("persistent_workers", False)),
            "prefetch_factor": training.get("prefetch_factor", None),
            "precision": precision,
            "tf32": bool(training.get("tf32", True)),
            "device_index": device_index,
            "max_steps": max_steps,
            "peak_gpu_memory_gb": round(peak_memory_gb, 4),
            "train_stats": train_stats,
            "seconds": round(time.time() - run_started, 4),
            "real_cier_training": True,
            "probe_only": True,
            "full_training_started": False,
            "updated_at": utc_now(),
        }
        write_json(run_path / "meta" / "stage_status.json", status)
        _update_run_summary(run_path, status=status)
        if args.probe_output:
            write_json(Path(args.probe_output), status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return status

    if checkpoint_dir.is_dir() and prompt_path.is_file():
        try:
            model.model.load_adapter(str(checkpoint_dir), adapter_name="best_lora", is_trainable=False)
            model.model.set_adapter("best_lora")
            best_adapter_loaded = True
        except Exception:
            best_adapter_loaded = False
        state = torch.load(str(prompt_path), map_location=device)
        model.prompt_encoder.load_state_dict(state)

    for split, dataset in (("valid", target_valid_dataset), ("test", target_test_dataset)):
        raw_rows = _predict_full(model, dataset, tokenizer, config, split=split)
        write_jsonl(run_path / "model" / f"cier_raw_{split}_predictions.jsonl", raw_rows)
    status = {
        "schema_version": "odcr_cier_stage_status_v1",
        "status": "trained",
        "baseline": BASELINE,
        "task_id": int(args.task),
        "run_id": args.run_id,
        "data_run_id": args.data_run_id or args.run_id,
        "mode": mode,
        "checkpoint": str(checkpoint_dir),
        "selected_checkpoint": str(checkpoint_dir),
        "prompt_encoder": str(prompt_path),
        "checkpoint_sha256": sha256_file(prompt_path) if prompt_path.is_file() else None,
        "selection_split": "target_valid",
        "best_valid_loss": best_loss,
        "best_adapter_loaded_for_prediction": best_adapter_loaded,
        "training_parameters": {
            "model_name": model_name,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "accumulation_steps": accumulation_steps,
            "global_batch": batch_size * accumulation_steps,
            "learning_rate": learning_rate,
            "num_workers": int(training.get("num_workers", 0)),
            "eval_num_workers": int(training.get("eval_num_workers", training.get("num_workers", 0))),
            "pin_memory": bool(training.get("pin_memory", True)),
            "persistent_workers": bool(training.get("persistent_workers", False)),
            "prefetch_factor": training.get("prefetch_factor", None),
            "precision": precision,
            "tf32": bool(training.get("tf32", True)),
            "device_index": device_index,
            "max_generation_length": word,
        },
        "train_stats": train_stats,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated(device) / (1024**3), 4),
        "seconds": round(time.time() - run_started, 4),
        "real_cier_training": True,
        "full_training_started": True,
        "updated_at": utc_now(),
    }
    write_json(run_path / "meta" / "stage_status.json", status)
    _update_run_summary(run_path, status=status)
    return status


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.task)
    config = _apply_training_overrides(config, args)
    mode = normalize_mode(args.mode, config)
    args.run_id = args.run_id or ("smoke" if args.smoke else "dry_run" if args.dry_run else "manual")
    run_path = run_dir(args.task, args.run_id)
    data_run_path = run_dir(args.task, args.data_run_id or args.run_id)
    if args.dry_run:
        plan = _training_plan(config, mode, run_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return plan
    ensure_run_layout(run_path)
    write_resolved_config(run_path / "meta" / "resolved_config.json", config, mode=mode, run_id=args.run_id)
    if args.smoke:
        return _run_smoke(args, config, mode, run_path)
    if args.predict_only:
        return _run_predict_only(args, config, mode, run_path, data_run_path)
    return _run_full(args, config, mode, run_path, data_run_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or smoke-test the external CIER-adapted ODCR baseline.")
    parser.add_argument("--task", type=int, required=True, choices=[2, 5, 7, 8])
    parser.add_argument("--mode", choices=["source_to_target", "target_only", "source-to-target"], default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-run-id", default=None, help="Read prepared CIER records from another run id.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--probe", action="store_true", help="Run a bounded real CUDA training probe without exporting predictions.")
    parser.add_argument("--probe-output", default=None)
    parser.add_argument("--predict-only", action="store_true", help="Load an existing best checkpoint and export raw CIER predictions.")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--eval-num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=None)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--source-epochs", type=int, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--show-train-loss-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.set_defaults(pin_memory=None, persistent_workers=None, tf32=None)
    args = parser.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
