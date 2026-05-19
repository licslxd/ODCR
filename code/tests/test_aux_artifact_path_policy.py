from __future__ import annotations

import json

import pytest

from odcr_core.aux.artifacts.path_policy import ArtifactPathPolicy, PathPolicyError, load_json_file, repo_relative_path


def test_path_policy_separates_formal_and_test_roots(tmp_path) -> None:
    policy = ArtifactPathPolicy(tmp_path)
    assert policy.formal_run_dir("step5", 2, "1") == tmp_path / "runs" / "step5" / "task2" / "1"
    assert policy.test_run_like_dir("step5", 2, "case") == tmp_path / "test_artifacts" / "runs_like" / "step5" / "task2" / "case"


def test_validation_cannot_write_formal_runs(tmp_path) -> None:
    policy = ArtifactPathPolicy(tmp_path)
    with pytest.raises(PathPolicyError):
        policy.assert_not_formal_for_validation(tmp_path / "runs" / "step5" / "task2" / "1")


def test_json_and_repo_relative_helpers_are_pure(tmp_path) -> None:
    payload_path = tmp_path / "meta" / "payload.json"
    payload_path.parent.mkdir()
    payload_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert load_json_file(payload_path) == {"ok": True}
    assert repo_relative_path(tmp_path, payload_path) == "meta/payload.json"
    assert repo_relative_path(tmp_path, None) is None
