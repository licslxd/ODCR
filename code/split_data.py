import argparse
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_contract import (
    PROCESSED_COLUMN_ORDER,
    SPLIT_COLUMN_ORDER,
    normalize_preprocess_dataframe,
    read_preprocess_csv,
    write_preprocess_csv,
)
from odcr_core.preprocess_metadata import split_policy_stats

DATASETS = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]


def _install_resolved_data_dir(data_dir: str) -> None:
    raw = str(data_dir or "").strip()
    if not raw:
        raise ValueError("split_data.py requires --data-dir from the resolved One-Control preprocess payload.")
    os.environ["ODCR_RESOLVED_DATA_DIR"] = os.path.abspath(os.path.expanduser(raw))


def _resolved_data_dir() -> str:
    raw = str(os.environ.get("ODCR_RESOLVED_DATA_DIR") or "").strip()
    if not raw:
        raise RuntimeError(
            "split_data.py requires ODCR_RESOLVED_DATA_DIR/--data-dir from ./odcr; "
            "refusing to read configs/odcr.yaml as a child-side fallback."
        )
    return os.path.abspath(os.path.expanduser(raw))


def _select_processed_contract(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_preprocess_dataframe(df)
    return normalized.loc[:, list(PROCESSED_COLUMN_ORDER)].copy()


def _select_split_contract(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_preprocess_dataframe(df, require_split_indices=True)
    return normalized.loc[:, list(SPLIT_COLUMN_ORDER)].reset_index(drop=True).copy()


def split_func_with_stats(df, random_seed=42):
    df = _select_processed_contract(df)
    user_dict = {user: idx for idx, user in enumerate(df["user"].unique())}
    item_dict = {item: idx for idx, item in enumerate(df["item"].unique())}

    df["user_idx"] = df["user"].map(user_dict)
    df["item_idx"] = df["item"].map(item_dict)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=random_seed)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_seed)
    valid_rows_before_filter = len(valid_df)
    test_rows_before_filter = len(test_df)
    valid_df = valid_df[
        valid_df["user_idx"].isin(train_df["user_idx"]) & valid_df["item_idx"].isin(train_df["item_idx"])
    ]
    test_df = test_df[
        test_df["user_idx"].isin(train_df["user_idx"]) & test_df["item_idx"].isin(train_df["item_idx"])
    ]
    train_df = _select_split_contract(train_df)
    valid_df = _select_split_contract(valid_df)
    test_df = _select_split_contract(test_df)
    stats = split_policy_stats(
        processed_rows=len(df),
        train_rows=len(train_df),
        valid_rows_after_filter=len(valid_df),
        test_rows_after_filter=len(test_df),
    )
    stats["valid_rows_before_filter"] = int(valid_rows_before_filter)
    stats["test_rows_before_filter"] = int(test_rows_before_filter)
    stats["valid_filtered_cold_user_item_rows"] = int(valid_rows_before_filter - len(valid_df))
    stats["test_filtered_cold_user_item_rows"] = int(test_rows_before_filter - len(test_df))
    stats["filtered_cold_user_item_rows"] = int(
        stats["valid_filtered_cold_user_item_rows"] + stats["test_filtered_cold_user_item_rows"]
    )
    stats["split_loss_rows"] = int(stats["filtered_cold_user_item_rows"])
    return train_df, valid_df, test_df, stats


def split_func(df, random_seed=42):
    train_df, valid_df, test_df, _stats = split_func_with_stats(df, random_seed=random_seed)
    return train_df, valid_df, test_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="processed.csv -> train.csv / valid.csv / test.csv；支持按数据集子集执行。",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="逗号分隔的数据集名，如 'Yelp' 或 'AM_Movies,TripAdvisor'；不传则处理全部。",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Resolved project.data_dir from ./odcr; direct YAML fallback is forbidden.",
    )
    return parser.parse_args()


def _resolve_datasets(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return list(DATASETS)
    datasets = [d.strip() for d in str(raw).split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        raise ValueError(f"未知数据集: {unknown}; 可选值: {DATASETS}")
    return datasets


if __name__ == "__main__":
    args = _parse_args()
    _install_resolved_data_dir(args.data_dir)
    for dataset in _resolve_datasets(args.datasets):
        df = read_preprocess_csv(os.path.join(_resolved_data_dir(), dataset, "processed.csv"))
        train_df, valid_df, test_df, stats = split_func_with_stats(df)
        print(f"{dataset}: train:{len(train_df)}, valid:{len(valid_df)}, test:{len(test_df)}")
        print(f"[split_data] dataset={dataset} split_policy_stats={json.dumps(stats, sort_keys=True)}")
        write_preprocess_csv(
            train_df,
            os.path.join(_resolved_data_dir(), dataset, "train.csv"),
            require_split_indices=True,
            column_order=SPLIT_COLUMN_ORDER,
        )
        write_preprocess_csv(
            valid_df,
            os.path.join(_resolved_data_dir(), dataset, "valid.csv"),
            require_split_indices=True,
            column_order=SPLIT_COLUMN_ORDER,
        )
        write_preprocess_csv(
            test_df,
            os.path.join(_resolved_data_dir(), dataset, "test.csv"),
            require_split_indices=True,
            column_order=SPLIT_COLUMN_ORDER,
        )
