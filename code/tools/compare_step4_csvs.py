# -*- coding: utf-8 -*-
"""
比较 3 个 Step4 相关 CSV 是否在优化前后生成结果一致。

用法:
  python code/tools/compare_step4_csvs.py --a p0.csv --b p1.csv --c p2.csv
  python code/tools/compare_step4_csvs.py --dir /path/to/run_dir
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TEXT_TRUNC = 120
DIFF_SAMPLE_LIMIT = 20
DEFAULT_NAMES = (
    "odcr_routing_train.csv",
    "odcr_routing_train1.csv",
    "odcr_routing_train2.csv",
)


def file_digest(path: str) -> Dict[str, Any]:
    """文件是否存在、大小、行数、MD5、SHA256（单遍流式读取）。"""
    out: Dict[str, Any] = {
        "path": path,
        "exists": False,
        "size_bytes": None,
        "line_count": None,
        "md5": None,
        "sha256": None,
        "error": None,
    }
    if not path or not os.path.isfile(path):
        out["error"] = "not a file" if path else "empty path"
        return out
    out["exists"] = True
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    nlines = 0
    size = 0
    try:
        with open(path, "rb") as f:
            for line in f:
                nlines += 1
                size += len(line)
                md5.update(line)
                sha.update(line)
        out["size_bytes"] = size
        out["line_count"] = nlines
        out["md5"] = md5.hexdigest()
        out["sha256"] = sha.hexdigest()
    except OSError as e:
        out["error"] = str(e)
    return out


def load_csv(path: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """使用 pandas 读取 CSV；失败返回 (None, error_msg)。"""
    try:
        df = pd.read_csv(path, low_memory=False)
        return df, None
    except Exception as e:  # noqa: BLE001 — 报告任意读盘/解析错误
        return None, str(e)


def _align_columns(left: pd.DataFrame, right: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按 left 的列顺序对齐 right；列集合不一致时仍返回尽力对齐的视图。"""
    cols = list(left.columns)
    missing = [c for c in cols if c not in right.columns]
    if missing:
        return left, right
    return left, right[cols].copy()


def compare_schema(
    dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """多文件 schema：列名集合、列顺序、dtypes（以第一个非空为基准列序）。"""
    keys = ["A", "B", "C"]
    ref = None
    ref_k = None
    for k in keys:
        if dfs.get(k) is not None and len(dfs[k].columns) > 0:
            ref = dfs[k]
            ref_k = k
            break
    if ref is None:
        return {
            "columns_equal": False,
            "column_order_equal": False,
            "dtypes_equal": False,
            "detail": "无法读取任一 DataFrame",
        }

    col_sets = {k: set(dfs[k].columns) for k in keys if dfs[k] is not None}
    columns_equal = len(set(frozenset(s) for s in col_sets.values())) <= 1
    order_equal = all(
        list(dfs[k].columns) == list(ref.columns) for k in keys if dfs[k] is not None
    )
    dtypes_equal = True
    dtype_mismatch: List[str] = []
    if columns_equal:
        for k in keys:
            dfk = dfs[k]
            if dfk is None:
                continue
            for c in ref.columns:
                if c not in dfk.columns:
                    dtypes_equal = False
                    dtype_mismatch.append(f"{k}: missing col {c!r}")
                    continue
                if ref[c].dtype != dfk[c].dtype:
                    dtypes_equal = False
                    dtype_mismatch.append(
                        f"{k}[{c!r}] {dfk[c].dtype} vs {ref_k}[{c!r}] {ref[c].dtype}"
                    )
    else:
        dtypes_equal = False

    return {
        "columns_equal": columns_equal,
        "column_order_equal": order_equal,
        "dtypes_equal": dtypes_equal,
        "dtype_mismatch": dtype_mismatch,
        "ref_key": ref_k,
    }


def _strict_series_equal(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """逐元素严格相等（NaN 与 NaN 视为相等）。"""
    s1 = s1.reset_index(drop=True)
    s2 = s2.reset_index(drop=True)
    return s1.eq(s2) | (s1.isna() & s2.isna())


def _truncate_for_display(val: Any, col: str) -> str:
    """差异展示用字符串；长文本截断。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "<NaN>"
    t = str(val)
    is_text_col = "explanation" in col.lower() or (
        len(t) > TEXT_TRUNC and not t.replace(".", "", 1).replace("-", "", 1).isdigit()
    )
    if len(t) > TEXT_TRUNC and (is_text_col or len(t) > 200):
        return t[:TEXT_TRUNC] + f"... ({len(t)} chars)"
    if len(t) > TEXT_TRUNC:
        return t[:TEXT_TRUNC] + "..."
    return t


def compare_content_strict(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str,
    name2: str,
) -> Dict[str, Any]:
    """严格内容比较：原始顺序、列对齐后顺序；缺失计数；dtype（按列名对齐后）。"""
    out: Dict[str, Any] = {
        "pair": (name1, name2),
        "rows_equal": len(df1) == len(df2),
        "cols_set_equal": set(df1.columns) == set(df2.columns),
        "ordered_equal": False,
        "aligned_equal": False,
        "na_counts_equal": False,
        "dtypes_pairwise_equal": True,
        "dtype_diffs": [],
    }
    out["strict_equal"] = False
    if not out["rows_equal"] or not out["cols_set_equal"]:
        return out

    out["ordered_equal"] = bool(df1.equals(df2))
    _, d2a = _align_columns(df1, df2)
    out["aligned_equal"] = bool(df1.equals(d2a))

    na1 = df1.isna().sum()
    na2 = df2.isna().sum()
    na2a = d2a.isna().sum()
    out["na_counts_equal"] = bool(na1.equals(na2a))

    dtype_diffs: List[str] = []
    for c in df1.columns:
        if df1[c].dtype != d2a[c].dtype:
            dtype_diffs.append(f"{c}: {name1}={df1[c].dtype} vs {name2}={d2a[c].dtype}")
    out["dtypes_pairwise_equal"] = len(dtype_diffs) == 0
    out["dtype_diffs"] = dtype_diffs

    out["strict_equal"] = bool(out["ordered_equal"] or out["aligned_equal"])
    return out


def compare_content_approx(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> Tuple[bool, List[str]]:
    """
    近似一致：float 列用 isclose；其余类型严格等于。
    列按 df1 顺序对齐 df2。
    """
    if len(df1) != len(df2) or set(df1.columns) != set(df2.columns):
        return False, ["shape or columns differ"]
    _, d2 = _align_columns(df1, df2)
    reasons: List[str] = []
    for c in df1.columns:
        s1 = df1[c].reset_index(drop=True)
        s2 = d2[c].reset_index(drop=True)
        if pd.api.types.is_float_dtype(s1) and pd.api.types.is_float_dtype(s2):
            v1 = pd.to_numeric(s1, errors="coerce")
            v2 = pd.to_numeric(s2, errors="coerce")
            m1 = v1.isna()
            m2 = v2.isna()
            if not (m1 == m2).all():
                reasons.append(f"{c}: NaN pattern differs")
                return False, reasons
            ok = np.ones(len(v1), dtype=bool)
            finite = ~m1
            if finite.any():
                ok[finite] = np.isclose(
                    v1.to_numpy()[finite],
                    v2.to_numpy()[finite],
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                )
            if not ok.all():
                reasons.append(f"{c}: float values differ beyond rtol/atol")
                return False, reasons
        elif pd.api.types.is_integer_dtype(s1) and pd.api.types.is_integer_dtype(s2):
            if not _strict_series_equal(s1, s2).all():
                reasons.append(f"{c}: integer/object mismatch")
                return False, reasons
        else:
            if not s1.equals(s2):
                reasons.append(f"{c}: non-float strict mismatch")
                return False, reasons
    return True, []


def compare_with_row_idx(
    dfs: Dict[str, pd.DataFrame],
    keys: Sequence[str] = ("A", "B", "C"),
) -> Dict[str, Any]:
    """row_idx：存在性、缺失、唯一性、稳定排序后两两严格相等。"""
    out: Dict[str, Any] = {
        "row_idx_exists": False,
        "per_file": {},
        "sorted_strict": {},
        "note": None,
    }
    first = None
    for k in keys:
        df = dfs.get(k)
        if df is None:
            continue
        first = df
        break
    if first is None or "row_idx" not in first.columns:
        out["note"] = "未找到 row_idx 列，跳过按 row_idx 的排序比较。"
        return out
    out["row_idx_exists"] = True
    for k in keys:
        df = dfs.get(k)
        if df is None:
            out["per_file"][k] = {"error": "no dataframe"}
            continue
        if "row_idx" not in df.columns:
            out["per_file"][k] = {"error": "missing row_idx"}
            continue
        r = df["row_idx"]
        missing = int(r.isna().sum())
        dup = int(r.duplicated().sum())
        uniq = bool(dup == 0)
        out["per_file"][k] = {
            "missing": missing,
            "duplicate_count": dup,
            "unique": uniq,
        }

    # 排序后比较：需三表均有 row_idx
    if not all(dfs.get(k) is not None and "row_idx" in dfs[k].columns for k in keys):
        out["note"] = "部分文件缺少 row_idx，无法做三表排序对齐比较。"
        return out

    def sort_stable(d: pd.DataFrame) -> pd.DataFrame:
        return d.sort_values("row_idx", kind="mergesort").reset_index(drop=True)

    sorted_map = {k: sort_stable(dfs[k]) for k in keys}
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for p, q in pairs:
        r1 = compare_content_strict(sorted_map[p], sorted_map[q], p, q)
        out["sorted_strict"][f"{p}_vs_{q}"] = r1.get("strict_equal", False)

    all_sorted = all(out["sorted_strict"].values())
    out["all_sorted_equal"] = all_sorted
    return out


def format_diff_report(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str,
    name2: str,
    max_samples: int = DIFF_SAMPLE_LIMIT,
) -> List[str]:
    """生成差异样例行（最多 max_samples 条），优先使用 row_idx 作为键。"""
    lines: List[str] = []
    if len(df1) != len(df2):
        lines.append(f"  [rows] {name1}={len(df1)} vs {name2}={len(df2)}")
        return lines
    if set(df1.columns) != set(df2.columns):
        only1 = set(df1.columns) - set(df2.columns)
        only2 = set(df2.columns) - set(df1.columns)
        lines.append(f"  [columns] only in {name1}: {sorted(only1)}")
        lines.append(f"  [columns] only in {name2}: {sorted(only2)}")
        return lines

    _, d2 = _align_columns(df1, df2)
    d1 = df1.reset_index(drop=True)
    d2 = d2.reset_index(drop=True)

    use_key = "row_idx" if "row_idx" in d1.columns else None
    samples = 0
    for col in d1.columns:
        if samples >= max_samples:
            break
        mask = ~_strict_series_equal(d1[col], d2[col])
        if not mask.any():
            continue
        bad_idx = mask[mask].index.tolist()
        lines.append(f"  [column {col!r}] {int(mask.sum())} cells differ, showing up to {max_samples - samples}:")
        for i in bad_idx:
            if samples >= max_samples:
                break
            row_label = f"row_idx={d1.loc[i, use_key]!r}" if use_key else f"position={i}"
            v1 = _truncate_for_display(d1.loc[i, col], col)
            v2 = _truncate_for_display(d2.loc[i, col], col)
            lines.append(f"    - {row_label} | {name1}={v1!s} | {name2}={v2!s}")
            samples += 1
    if not lines:
        lines.append("  (无单元级差异明细，可能为 dtype/索引级别差异)")
    return lines


def _pair_key(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def run_comparison(paths: Dict[str, str]) -> int:
    labels = ("A", "B", "C")
    print("=" * 50)
    print("CSV Comparison Summary")
    print("=" * 50)
    for k in labels:
        print(f"{k}: {paths[k]}")
    print()

    digests = {k: file_digest(paths[k]) for k in labels}
    print("[File-level]")
    for k in labels:
        d = digests[k]
        ex = d["exists"]
        print(f"- {k} exists: {ex}")
        if not ex:
            print(f"  error: {d.get('error')}")
            continue
        print(f"  size_bytes: {d['size_bytes']}")
        print(f"  line_count: {d['line_count']}")
        print(f"  md5: {d['md5']}")
        print(f"  sha256: {d['sha256']}")
    print()

    dfs: Dict[str, Optional[pd.DataFrame]] = {}
    load_err: Dict[str, str] = {}
    for k in labels:
        if not digests[k]["exists"]:
            dfs[k] = None
            load_err[k] = "file missing"
            continue
        df, err = load_csv(paths[k])
        dfs[k] = df  # type: ignore[assignment]
        load_err[k] = err or ""

    for k in labels:
        if load_err[k]:
            print(f"[Load] {k}: FAILED — {load_err[k]}")
    if any(load_err[k] for k in labels):
        print()
        print("[Conclusion]")
        print("- files differ in actual content (无法读取部分 CSV)")
        return 2

    # 此时 dfs 均为非 None
    dfA, dfB, dfC = dfs["A"], dfs["B"], dfs["C"]
    assert dfA is not None and dfB is not None and dfC is not None

    print("[DataFrame rows/cols]")
    for k in labels:
        dfk = dfs[k]
        assert dfk is not None
        print(f"- {k}: rows={len(dfk)} cols={len(dfk.columns)}")
    print()

    schema = compare_schema(dfs)  # type: ignore[arg-type]
    print("[Schema]")
    print(f"- columns_equal: {schema['columns_equal']}")
    print(f"- column_order_equal: {schema['column_order_equal']}")
    print(f"- dtypes_equal: {schema['dtypes_equal']}")
    if schema.get("dtype_mismatch"):
        for s in schema["dtype_mismatch"][:15]:
            print(f"  dtype note: {s}")
        if len(schema["dtype_mismatch"]) > 15:
            print(f"  ... ({len(schema['dtype_mismatch'])} total dtype notes)")
    print()

    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    strict_results: Dict[str, Dict[str, Any]] = {}
    print("[Strict content equality]")
    for p, q in pairs:
        r = compare_content_strict(dfs[p], dfs[q], p, q)  # type: ignore[index]
        strict_results[_pair_key(p, q)] = r
        eq = r.get("strict_equal", False)
        print(f"- {p} vs {q}: {eq}")
        if not r["rows_equal"]:
            print(f"    (row count: {p}={len(dfs[p])} vs {q}={len(dfs[q])})")  # type: ignore[arg-type]
        if not r["cols_set_equal"]:
            print(f"    (column set differs)")
        elif not r["ordered_equal"] and r["aligned_equal"]:
            print(f"    (note: 列顺序不同但列名对齐后内容一致)")
        elif not r["na_counts_equal"]:
            print(f"    (note: 缺失值计数不一致)")
        elif not r["dtypes_pairwise_equal"]:
            for t in r["dtype_diffs"][:5]:
                print(f"    dtype: {t}")
    print()

    approx_results: Dict[str, bool] = {}
    print("[Approx numeric equality] (float: rtol=1e-6, atol=1e-8)")
    for p, q in pairs:
        ok, _rs = compare_content_approx(dfs[p], dfs[q])  # type: ignore[arg-type]
        approx_results[_pair_key(p, q)] = ok
        print(f"- {p} vs {q}: {ok}")
    all_strict_pairs = all(strict_results[_pair_key(p, q)]["strict_equal"] for p, q in pairs)
    all_approx_pairs = all(approx_results[_pair_key(p, q)] for p, q in pairs)
    print()
    print("[Equality flags]")
    print(f"- strict_equal (all pairs): {all_strict_pairs}")
    print(f"- approx_equal (all pairs): {all_approx_pairs}")
    print()

    ridx = compare_with_row_idx(dfs)  # type: ignore[arg-type]
    print("[row_idx checks]")
    print(f"- row_idx exists: {ridx['row_idx_exists']}")
    if ridx.get("note"):
        print(f"  note: {ridx['note']}")
    if ridx["row_idx_exists"]:
        for k in labels:
            info = ridx["per_file"].get(k, {})
            if "error" in info:
                print(f"  {k}: {info['error']}")
            else:
                print(
                    f"  {k}: unique={info.get('unique')} missing={info.get('missing')} "
                    f"duplicates={info.get('duplicate_count')}"
                )
        if ridx.get("sorted_strict"):
            for sk, sv in ridx["sorted_strict"].items():
                print(f"  sorted strict equal [{sk}]: {sv}")
            print(f"  all_sorted_equal: {ridx.get('all_sorted_equal', False)}")
    print()

    # 差异报告（若存在不严格相等的对）
    need_diff = any(not strict_results[_pair_key(p, q)] for p, q in pairs)
    if need_diff:
        print("[Difference samples] (max 20 total across pairs)")
        remaining = DIFF_SAMPLE_LIMIT
        for p, q in pairs:
            if remaining <= 0:
                break
            if strict_results[_pair_key(p, q)].get("strict_equal"):
                continue
            print(f"--- {p} vs {q} ---")
            lines = format_diff_report(
                dfs[p], dfs[q], p, q, max_samples=remaining  # type: ignore[arg-type]
            )
            for ln in lines:
                print(ln)
            remaining -= sum(1 for x in lines if x.startswith("    - "))
        print()

    # Conclusion
    all_strict = all_strict_pairs
    all_approx = all_approx_pairs
    sha_all_same = len({digests[k]["sha256"] for k in labels if digests[k]["exists"]}) == 1

    print("[Conclusion]")
    if all_strict and sha_all_same:
        print("- three files are exactly identical")
        return 0
    if all_strict and not sha_all_same:
        print("- three files are logically identical (strict equal) but binary hashes differ (换行/编码等)")
        return 0
    if (
        (not all_strict)
        and ridx.get("row_idx_exists")
        and ridx.get("all_sorted_equal")
    ):
        print(
            "- three files match after stable sort by row_idx "
            "(原始行序或逐行对齐方式不同，但按 row_idx 排序后两两严格一致)"
        )
        return 0
    if (not all_strict) and all_approx:
        print("- files differ only in float precision (strict=False, approx=True)；仅浮点微小误差，不影响语义")
        return 1
    print("- files differ in actual content")
    return 2


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="比较三个 Step4 正式训练表 odcr_routing_train*.csv 是否一致。",
    )
    p.add_argument("--a", default=None, help="第一个 CSV 路径")
    p.add_argument("--b", default=None, help="第二个 CSV 路径")
    p.add_argument("--c", default=None, help="第三个 CSV 路径")
    p.add_argument(
        "--dir",
        default=None,
        help=f"目录模式：比较其下 {', '.join(DEFAULT_NAMES)}",
    )
    args = p.parse_args(argv)
    if args.dir:
        base = Path(args.dir).expanduser().resolve()
        args.a = str(base / DEFAULT_NAMES[0])
        args.b = str(base / DEFAULT_NAMES[1])
        args.c = str(base / DEFAULT_NAMES[2])
    else:
        if not (args.a and args.b and args.c):
            p.error("请提供 --dir，或同时提供 --a、--b、--c")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    paths = {"A": args.a, "B": args.b, "C": args.c}
    return run_comparison(paths)


if __name__ == "__main__":
    sys.exit(main())
