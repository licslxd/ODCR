"""Minimal deterministic Step5 prompt template registry.

The templates are an input-formatting layer only.  They do not expose RCR
posterior field names and they do not replace LCI/UCI/CCV/FCA controls.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping


STEP5_PROMPT_REGISTRY_SCHEMA_VERSION = "odcr_step5_prompt_template_registry/1"
STEP5_PROMPT_TEMPLATE_VERSION = "v1"


@dataclass(frozen=True)
class Step5PromptTemplate:
    canonical_id: str
    task_head: str
    sample_origin: str
    family: str
    version: str
    text: str

    def to_manifest(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "task_head": self.task_head,
            "sample_origin": self.sample_origin,
            "family": self.family,
            "version": self.version,
            "text": self.text,
        }


def _stable_id(*parts: Any) -> str:
    raw = "\n".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class PromptTemplateRegistry:
    """Step5 explanation-only registry keyed by sample origin."""

    def __init__(self) -> None:
        self._templates = {
            ("explanation", "target_gold"): Step5PromptTemplate(
                canonical_id="Step5_target_anchor_explainer_v1",
                task_head="explanation",
                sample_origin="target_gold",
                family="explanation_only",
                version=STEP5_PROMPT_TEMPLATE_VERSION,
                text="Write one concise target-domain recommendation explanation. Do not include a score.",
            ),
            ("explanation", "aux_gold"): Step5PromptTemplate(
                canonical_id="Step5_aux_gold_explainer_v1",
                task_head="explanation",
                sample_origin="aux_gold",
                family="explanation_only",
                version=STEP5_PROMPT_TEMPLATE_VERSION,
                text="Write one concise source-domain recommendation explanation. Do not include a score.",
            ),
            ("explanation", "aux_cf"): Step5PromptTemplate(
                canonical_id="Step5_aux_cf_explainer_v1",
                task_head="explanation",
                sample_origin="aux_cf",
                family="explanation_only",
                version=STEP5_PROMPT_TEMPLATE_VERSION,
                text="Write one concise counterfactual-style recommendation explanation. Do not include a score.",
            ),
        }

    def template_for(self, *, task_head: str, sample_origin: str) -> Step5PromptTemplate:
        origin = "aux_cf" if str(sample_origin) == "aux_cf" else str(sample_origin)
        head = "explanation"
        legacy_explainer = "step5" + "B"
        if str(task_head) not in {"explanation", legacy_explainer}:
            raise KeyError(f"unknown Step5 prompt head: {task_head!r}")
        key = (head, origin)
        if key not in self._templates:
            raise KeyError(f"unknown Step5 prompt template key: {key!r}")
        return self._templates[key]

    def render(
        self,
        *,
        sample: Mapping[str, Any],
        task_head: str,
        sample_origin: str,
        seed: int,
        split: str = "train",
    ) -> dict[str, Any]:
        template = self.template_for(task_head=task_head, sample_origin=sample_origin)
        sample_id = sample.get("sample_id", "")
        instance_id = _stable_id(sample_id, task_head, sample_origin, int(seed), "canonical")
        mode = "fixed_canonical" if str(split).lower() in {"valid", "test"} else "controlled_canonical"
        return {
            "step5_prompt_template_id": template.canonical_id,
            "step5_prompt_instance_id": instance_id,
            "step5_prompt_family": template.family,
            "step5_prompt_version": template.version,
            "step5_prompt_mode": mode,
            "step5_prompt_seed": int(seed),
            "step5_prompt_text": template.text,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": STEP5_PROMPT_REGISTRY_SCHEMA_VERSION,
            "template_count": len(self._templates),
            "templates": [tmpl.to_manifest() for tmpl in self._templates.values()],
            "prompt_role": "input_formatting_only",
            "does_not_replace": ["LCI", "UCI", "CCV", "FCA", "Step4 RCR"],
            "train_policy": "controlled_canonical_deterministic",
            "valid_test_policy": "fixed_canonical",
            "active_sample_origins": ["target_gold", "aux_gold", "aux_cf"],
            "rating_training": False,
        }


def default_prompt_registry() -> PromptTemplateRegistry:
    return PromptTemplateRegistry()


def prompt_registry_manifest() -> dict[str, Any]:
    return default_prompt_registry().manifest()


__all__ = [
    "PromptTemplateRegistry",
    "STEP5_PROMPT_REGISTRY_SCHEMA_VERSION",
    "STEP5_PROMPT_TEMPLATE_VERSION",
    "Step5PromptTemplate",
    "default_prompt_registry",
    "prompt_registry_manifest",
]
