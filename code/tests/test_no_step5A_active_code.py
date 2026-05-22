from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
NEEDLES = [
    "Step5" + "A",
    "step5" + "A",
    "step5" + "a",
    "Step5" + "B",
    "step5" + "B",
    "Retired" + "ScorerBranch",
    "retired" + "_scorer_branch",
    "latest_step5" + "A",
    "latest_" + "retired" + "_scorer_branch",
    "step5" + "A_rating",
    "A_target_gold_" + "scorer",
    "A_aux_gold_" + "scorer",
    "A_aux_cf_" + "scorer",
    "combined_step5_" + "ready",
    "paired_" + "status",
    "combined_" + "head",
    "combined " + "head",
    "Step5" + "A-Small",
    "rating-" + "only",
    "teacher_" + "parity",
    "frozen_" + "teacher",
    "residual_" + "calibration",
    "step5_rating_" + "handoff",
    "step5_rating_" + "quality",
    "step5_multiseed_" + "rating",
]
PATTERN = re.compile("|".join(re.escape(x) for x in NEEDLES))


def test_no_forbidden_rating_branch_strings_in_active_tree() -> None:
    roots = [ROOT / "code", ROOT / "configs", ROOT / "docs", ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "odcr"]
    hits: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for path in paths:
            if any(part in {"__pycache__", ".pytest_cache"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if PATTERN.search(text):
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []
