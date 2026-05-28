from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / "odcr"), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=90,
    )


def test_step5_help_does_not_offer_retired_rating_head() -> None:
    proc = _run("step5", "--help")
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0
    assert "finalize-rating-handoff" not in output
    assert "rating-quality-diagnostic" not in output
    assert "multiseed-rating" not in output
    assert ("step5" + "A") not in output
    assert "combined" not in output


def test_retired_rating_head_fails_fast() -> None:
    proc = _run("step5", "--task", "2", "--from-step4-run", "1", "--head", "step5" + "A", "--dry-run")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "rating now uses rating_source" in output
    assert "explanations only" in output


def test_legacy_step5b_head_fails_fast() -> None:
    proc = _run("step5", "--task", "2", "--from-step4-run", "1", "--head", "step5" + "B", "--dry-run")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "rating now uses rating_source" in output
    assert "explanations only" in output


def test_legacy_grouped_head_fails_fast() -> None:
    proc = _run("step5", "--task", "2", "--from-step4-run", "1", "--head", "combined", "--dry-run")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "rating now uses rating_source" in output
    assert "explanations only" in output


def test_retired_step4_dedicated_export_alias_is_not_user_reachable() -> None:
    proc = _run("step4", "export-step5-dedicated", "--help")
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "export-step5-dedicated" in output
    assert "invalid choice" in output
