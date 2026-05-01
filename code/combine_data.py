import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import MERGED_COLUMN_ORDER, read_preprocess_csv, write_preprocess_csv

TASK_ID_TO_DOMAINS = {
    1: ("AM_Electronics", "AM_CDs"),
    2: ("AM_Movies", "AM_CDs"),
    3: ("AM_CDs", "AM_Electronics"),
    4: ("AM_Movies", "AM_Electronics"),
    5: ("AM_CDs", "AM_Movies"),
    6: ("AM_Electronics", "AM_Movies"),
    7: ("Yelp", "TripAdvisor"),
    8: ("TripAdvisor", "Yelp"),
}


def _install_resolved_roots(*, data_dir: str, merged_dir: str) -> None:
    data_raw = str(data_dir or "").strip()
    merged_raw = str(merged_dir or "").strip()
    if not data_raw or not merged_raw:
        raise ValueError("combine_data.py requires --data-dir and --merged-dir from the resolved preprocess payload.")
    os.environ["ODCR_RESOLVED_DATA_DIR"] = os.path.abspath(os.path.expanduser(data_raw))
    os.environ["ODCR_RESOLVED_MERGED_DIR"] = os.path.abspath(os.path.expanduser(merged_raw))


def _resolved_data_dir() -> str:
    raw = str(os.environ.get("ODCR_RESOLVED_DATA_DIR") or "").strip()
    if not raw:
        raise RuntimeError(
            "combine_data.py requires ODCR_RESOLVED_DATA_DIR/--data-dir from ./odcr; "
            "refusing to read configs/odcr.yaml as a child-side fallback."
        )
    return os.path.abspath(os.path.expanduser(raw))


def _resolved_merged_dir() -> str:
    raw = str(os.environ.get("ODCR_RESOLVED_MERGED_DIR") or "").strip()
    if not raw:
        raise RuntimeError(
            "combine_data.py requires ODCR_RESOLVED_MERGED_DIR/--merged-dir from ./odcr; "
            "refusing to read configs/odcr.yaml as a child-side fallback."
        )
    return os.path.abspath(os.path.expanduser(raw))


def _resolve_task_pair(task_id: int) -> tuple[str, str]:
    try:
        return TASK_ID_TO_DOMAINS[int(task_id)]
    except KeyError as exc:
        valid = sorted(TASK_ID_TO_DOMAINS)
        raise ValueError(f"未知 task-id={task_id}; 可选值: {valid}") from exc


def _load_split_csv(dataset: str, split_name: str) -> pd.DataFrame:
    path = os.path.join(_resolved_data_dir(), dataset, f"{split_name}.csv")
    return read_preprocess_csv(path, require_split_indices=True)


def merge_task(task_id: int) -> None:
    source, target = _resolve_task_pair(task_id)

    source_train_data = _load_split_csv(source, "train").copy()
    target_train_data = _load_split_csv(target, "train").copy()
    source_valid_data = _load_split_csv(source, "valid").copy()
    target_valid_data = _load_split_csv(target, "valid").copy()

    max_user_idx = target_train_data["user_idx"].max()
    max_item_idx = target_train_data["item_idx"].max()

    source_train_data["user_idx"] += max_user_idx + 1
    source_train_data["item_idx"] += max_item_idx + 1
    source_valid_data["user_idx"] += max_user_idx + 1
    source_valid_data["item_idx"] += max_item_idx + 1

    source_train_data["domain"] = "auxiliary"
    target_train_data["domain"] = "target"
    source_valid_data["domain"] = "auxiliary"
    target_valid_data["domain"] = "target"

    merged_columns = list(MERGED_COLUMN_ORDER)
    merged_train_data = pd.concat(
        [source_train_data[merged_columns], target_train_data[merged_columns]],
        ignore_index=True,
    )
    merged_valid_data = pd.concat(
        [source_valid_data[merged_columns], target_valid_data[merged_columns]],
        ignore_index=True,
    )
    merged_train_data = merged_train_data.loc[:, merged_columns].reset_index(drop=True)
    merged_valid_data = merged_valid_data.loc[:, merged_columns].reset_index(drop=True)

    output_dir = os.path.join(_resolved_merged_dir(), str(task_id))
    os.makedirs(output_dir, exist_ok=True)
    merged_train_file = os.path.join(output_dir, "aug_train.csv")
    merged_valid_file = os.path.join(output_dir, "aug_valid.csv")
    write_preprocess_csv(
        merged_train_data,
        merged_train_file,
        require_split_indices=True,
        require_domain=True,
        column_order=MERGED_COLUMN_ORDER,
    )
    write_preprocess_csv(
        merged_valid_data,
        merged_valid_file,
        require_split_indices=True,
        require_domain=True,
        column_order=MERGED_COLUMN_ORDER,
    )

    print(
        f"[combine_data] task_id={task_id} source={source} target={target} "
        f"saved_train={merged_train_file}"
    )
    print(
        f"[combine_data] task_id={task_id} source={source} target={target} "
        f"saved_valid={merged_valid_file}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 task-id 生成 merged/<task>/aug_train.csv 与 aug_valid.csv；一次调用只生成一个 task。",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        metavar="N",
        help="combine 任务号，当前可选值为 1..8。",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Resolved project.data_dir from ./odcr; direct YAML fallback is forbidden.",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        required=True,
        help="Resolved project.merged_dir from ./odcr; direct YAML fallback is forbidden.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _install_resolved_roots(data_dir=args.data_dir, merged_dir=args.merged_dir)
    merge_task(int(args.task_id))


if __name__ == "__main__":
    main()
