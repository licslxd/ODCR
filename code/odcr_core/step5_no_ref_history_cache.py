from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCHEMA_VERSION = "odcr_step5_no_ref_history_cache/6_evidence_v3_phrase_quality"
BUILDER_VERSION = "step5_no_ref_history_cache/5_evidence_v3_phrase_quality"
NEUTRAL_CORE_NO_REF = "neutral_core_no_ref"
SELECTED_HISTORY_LITE_NO_REF = "selected_history_lite_no_ref"
TEXT_CLEAN_ITEM_ONLY_NO_REF = "text_clean_item_only_no_ref"
TEXT_CLEAN_ITEM_USER_NO_REF = "text_clean_item_user_no_ref"
TARGET_DOMAIN_PHRASE_NO_REF = "target_domain_phrase_no_ref"
CONTRASTIVE_EVIDENCE_NO_REF = "contrastive_evidence_no_ref"
ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF = "route_weighted_item_phrase_v2_no_ref"
SOFT_PREFIX_LIGHT_NO_REF = "soft_prefix_light_no_ref"
ALLOWED_INPUT_PROTOCOLS = (
    NEUTRAL_CORE_NO_REF,
    TEXT_CLEAN_ITEM_ONLY_NO_REF,
    TEXT_CLEAN_ITEM_USER_NO_REF,
    ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF,
    SOFT_PREFIX_LIGHT_NO_REF,
)
RETIRED_INPUT_PROTOCOLS = (
    SELECTED_HISTORY_LITE_NO_REF,
    TARGET_DOMAIN_PHRASE_NO_REF,
    CONTRASTIVE_EVIDENCE_NO_REF,
)
DEFAULT_NEUTRAL_EVIDENCE = "keywords none ; aspects none ; entities none"
DEFAULT_ENCODER_CONTENT_TOKEN_BUDGET = 96
DEFAULT_FALLBACK_ENCODER_CONTENT_TOKEN_BUDGET = 96

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "too",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
    "about",
    "after",
    "again",
    "all",
    "also",
    "any",
    "because",
    "could",
    "did",
    "does",
    "don",
    "get",
    "got",
    "just",
    "like",
    "more",
    "most",
    "much",
    "one",
    "only",
    "place",
    "really",
    "said",
    "see",
    "than",
    "then",
    "these",
    "those",
    "very",
    "would",
}

_LOW_VALUE_STANDALONE = {
    "good",
    "great",
    "nice",
    "excellent",
    "food",
    "place",
    "restaurant",
    "service",
    "yelp",
    "very",
    "really",
    "always",
    "just",
    "little",
    "like",
    "thing",
    "things",
    "much",
    "can",
    "would",
    "could",
    "not",
    "n't",
    "n",
    "t",
}
_SENTIMENT_SINGLETONS = {
    "good",
    "great",
    "nice",
    "excellent",
    "amazing",
    "awesome",
    "best",
    "delicious",
    "perfect",
    "bad",
    "terrible",
    "horrible",
    "awful",
    "fine",
    "love",
    "liked",
    "likes",
}
_GENERIC_ASPECT_NOUNS = {"food", "place", "restaurant", "service", "staff"}
_GENERIC_BAD_PHRASES = {
    "good food",
    "great food",
    "nice food",
    "excellent food",
    "good service",
    "great service",
    "nice service",
    "excellent service",
    "good place",
    "great place",
    "nice place",
    "yelp restaurant",
    "this place",
    "this restaurant",
}
_PHRASE_BAD_EDGE_TOKENS = {
    "also",
    "another",
    "anything",
    "au",
    "back",
    "being",
    "both",
    "came",
    "come",
    "comes",
    "coming",
    "did",
    "does",
    "doing",
    "done",
    "each",
    "else",
    "every",
    "everything",
    "getting",
    "gets",
    "go",
    "goes",
    "going",
    "gone",
    "how",
    "keep",
    "keeps",
    "lot",
    "make",
    "makes",
    "making",
    "many",
    "maybe",
    "now",
    "often",
    "other",
    "others",
    "out",
    "own",
    "part",
    "parts",
    "quite",
    "re",
    "same",
    "seem",
    "seems",
    "several",
    "should",
    "some",
    "something",
    "sometimes",
    "still",
    "sure",
    "take",
    "takes",
    "taking",
    "taste",
    "tastes",
    "tasting",
    "think",
    "things",
    "though",
    "thought",
    "usually",
    "want",
    "wants",
    "way",
    "what",
    "whatever",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "why",
    "will",
}
_PHRASE_BAD_INTERNAL_TOKENS = {
    "both",
    "every",
    "how",
    "re",
    "what",
    "whatever",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "whose",
    "why",
}
_PHRASE_BAD_PAIRS = {
    ("both", "medium"),
    ("drink", "twice"),
    ("drinks", "late"),
    ("drinks", "twice"),
    ("server", "every"),
    ("server", "time"),
    ("servers", "every"),
    ("servers", "time"),
}
_GENERIC_SENTIMENT_HEAD_NOUNS = _GENERIC_ASPECT_NOUNS | {
    "bar",
    "beer",
    "coffee",
    "drink",
    "drinks",
    "menu",
    "meal",
    "meals",
    "product",
    "products",
    "server",
    "servers",
    "taste",
    "tastes",
}
_CONCRETE_HINTS = {
    "ambiance",
    "atmosphere",
    "baked",
    "bakery",
    "bar",
    "bbq",
    "beans",
    "beer",
    "breakfast",
    "brunch",
    "bun",
    "buns",
    "burger",
    "burgers",
    "burrito",
    "cafe",
    "cake",
    "chicken",
    "coffee",
    "crab",
    "crispy",
    "crust",
    "dining",
    "dinner",
    "drink",
    "drinks",
    "fries",
    "friendly",
    "fresh",
    "garlic",
    "hotel",
    "indian",
    "italian",
    "juicy",
    "lunch",
    "macarons",
    "macrons",
    "menu",
    "noodles",
    "patio",
    "pizza",
    "portion",
    "portions",
    "price",
    "prices",
    "quick",
    "ramen",
    "reasonable",
    "roll",
    "salsa",
    "sandwich",
    "sauce",
    "slow",
    "spicy",
    "staff",
    "steak",
    "sushi",
    "sweet",
    "table",
    "tacos",
    "thai",
    "toasted",
    "wait",
    "wine",
}
_GOOD_SENTIMENT_NOUNS = {"value", "price", "prices", "deal", "deals"}
_NEGATIVE_KEEP_ADJECTIVES = {"slow", "long", "crowded", "rude", "cold", "dry"}


def stable_json_hash(payload: Any, *, length: int = 40) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: os.PathLike[str] | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False, "sha256": "", "size": 0}
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(p), "exists": True, "sha256": h.hexdigest(), "size": int(p.stat().st_size)}


def normalise_no_ref_evidence_config(
    raw: Mapping[str, Any] | None,
    *,
    neutral_content_evidence: str = DEFAULT_NEUTRAL_EVIDENCE,
) -> dict[str, Any]:
    cfg = dict(raw or {})
    def int_cfg(name: str, default: int, *, min_value: int) -> int:
        return max(min_value, int(cfg[name]) if name in cfg else int(default))

    protocol = str(cfg.get("input_protocol") or TEXT_CLEAN_ITEM_ONLY_NO_REF).strip()
    if protocol in RETIRED_INPUT_PROTOCOLS:
        raise ValueError(
            f"retired Step5 no-ref input_protocol {protocol!r}; use "
            f"{TEXT_CLEAN_ITEM_ONLY_NO_REF!r}, {TEXT_CLEAN_ITEM_USER_NO_REF!r}, "
            f"or {ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF!r}"
        )
    if protocol not in ALLOWED_INPUT_PROTOCOLS:
        raise ValueError(f"unsupported Step5 no-ref input_protocol: {protocol!r}")
    encoder_budget = max(4, int(cfg.get("encoder_content_token_budget") or DEFAULT_ENCODER_CONTENT_TOKEN_BUDGET))
    fallback_budget = max(4, int(cfg.get("fallback_encoder_content_token_budget") or DEFAULT_FALLBACK_ENCODER_CONTENT_TOKEN_BUDGET))
    if fallback_budget > encoder_budget:
        raise ValueError("fallback_encoder_content_token_budget must not exceed encoder_content_token_budget")
    return {
        "schema_version": str(cfg.get("schema_version") or "odcr_step5_no_ref_evidence_config/4_evidence_v3_phrase"),
        "input_protocol": protocol,
        "cache_namespace": str(cfg.get("cache_namespace") or "step5_no_ref_history"),
        "train_scope": str(cfg.get("train_scope") or "selected_effective_epoch"),
        "selected_effective_epochs": max(1, int(cfg.get("selected_effective_epochs") or 1)),
        "context_label": str(cfg.get("context_label", "") or "").strip(),
        "concise_prompt": str(
            cfg.get("concise_prompt") or "Write one concise review reason using the evidence."
        ).strip(),
        "phrase_prompt": str(
            cfg.get("phrase_prompt") or "Write one concise review reason using the evidence."
        ).strip(),
        "encoder_content_token_budget": int(encoder_budget),
        "fallback_encoder_content_token_budget": int(fallback_budget),
        "evidence_top_k": int_cfg("evidence_top_k", 6, min_value=1),
        "user_history_top_k": int_cfg("user_history_top_k", 6, min_value=0),
        "item_history_top_k": int_cfg("item_history_top_k", 6, min_value=0),
        "domain_prior_top_n": int_cfg("domain_prior_top_n", 128, min_value=1),
        "domain_prior_display_top_k": int_cfg("domain_prior_display_top_k", 0, min_value=0),
        "min_df": int_cfg("min_df", 3, min_value=1),
        "stopword_policy": str(cfg.get("stopword_policy") or "english_core_v1"),
        "term_token_limit": max(8, int(cfg.get("term_token_limit") or 64)),
        "neutral_content_evidence": str(cfg.get("neutral_content_evidence") or neutral_content_evidence),
        "smoke_cache_identity_includes_max_rows": bool(cfg.get("smoke_cache_identity_includes_max_rows", True)),
    }


def _normalised_tokens(text: Any, *, limit: int | None = None) -> list[str]:
    raw = str(text or "").lower().replace("n’t", " not ").replace("n't", " not ")
    out: list[str] = []
    for tok in _TOKEN_RE.findall(raw):
        tok = tok.strip("'_-")
        if not tok or tok in {"n", "t"}:
            continue
        if tok.isdigit():
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        out.append(tok)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def tokenize_history_text(text: Any) -> list[str]:
    out: list[str] = []
    for tok in _normalised_tokens(text):
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS or tok in _LOW_VALUE_STANDALONE:
            continue
        out.append(tok)
    return out


def _phrase_tokens(text: Any, *, token_limit: int) -> list[str]:
    return _normalised_tokens(text, limit=max(4, int(token_limit)))


def _is_fragment_token(tok: str) -> bool:
    token = str(tok or "").strip().lower()
    if len(token) < 2:
        return True
    if token in {"not", "n't", "n", "t"}:
        return True
    if token in _STOPWORDS:
        return True
    if token.isdigit() or not any(ch.isalpha() for ch in token):
        return True
    return False


def _phrase_generic_penalty(tokens: Sequence[str]) -> float:
    penalty = 0.0
    for tok in tokens:
        if tok in _LOW_VALUE_STANDALONE:
            penalty += 1.2
        if tok in _SENTIMENT_SINGLETONS:
            penalty += 0.7
    if "yelp" in tokens:
        penalty += 10.0
    return penalty


def _phrase_reject_reason(tokens: Sequence[str]) -> str | None:
    toks = [str(t or "").strip().lower() for t in tokens if str(t or "").strip()]
    if len(toks) < 2 or len(toks) > 4:
        return "length_not_2_to_4"
    if any(_is_fragment_token(tok) for tok in toks):
        return "fragment_or_stopword"
    phrase = " ".join(toks)
    if phrase in _GENERIC_BAD_PHRASES:
        return "generic_phrase"
    if "yelp" in toks:
        return "context_label"
    if any(tok in _PHRASE_BAD_INTERNAL_TOKENS for tok in toks):
        return "bad_internal_token"
    if toks[0] in _PHRASE_BAD_EDGE_TOKENS or toks[-1] in _PHRASE_BAD_EDGE_TOKENS:
        return "bad_edge_token"
    if len(toks) == 2 and (toks[0], toks[1]) in _PHRASE_BAD_PAIRS:
        return "weak_bigram"
    if any((left, right) in _PHRASE_BAD_PAIRS for left, right in zip(toks, toks[1:])):
        return "weak_adjacent_pair"
    if any(left in _SENTIMENT_SINGLETONS and right in _SENTIMENT_SINGLETONS for left, right in zip(toks, toks[1:])):
        return "sentiment_adjacent"
    if len(toks) == 2 and toks[0] in {"drink", "drinks", "server", "servers"} and toks[1] in {
        "late",
        "medium",
        "time",
        "twice",
    }:
        return "generic_event_phrase"
    for left, right in zip(toks, toks[1:]):
        if left in _SENTIMENT_SINGLETONS and right in _GENERIC_SENTIMENT_HEAD_NOUNS:
            return None if right in _GOOD_SENTIMENT_NOUNS else "sentiment_generic_head"
    concrete = [tok for tok in toks if tok not in _LOW_VALUE_STANDALONE and tok not in _STOPWORDS]
    if not concrete:
        return "no_concrete_token"
    if len(toks) == 2 and toks[0] in _SENTIMENT_SINGLETONS and toks[1] in _GENERIC_SENTIMENT_HEAD_NOUNS:
        return None if toks[1] in _GOOD_SENTIMENT_NOUNS else "sentiment_generic_bigram"
    if len(toks) == 2 and toks[1] in _PHRASE_BAD_EDGE_TOKENS:
        return "bad_tail_token"
    if all(tok in _LOW_VALUE_STANDALONE or tok in _SENTIMENT_SINGLETONS for tok in toks):
        return "sentiment_only"
    return None


def _valid_phrase_tokens(tokens: Sequence[str]) -> bool:
    return _phrase_reject_reason(tokens) is None


def collect_phrase_filter_audit(text: Any, *, token_limit: int = 64, max_examples: int = 24) -> list[dict[str, Any]]:
    toks = _phrase_tokens(text, token_limit=token_limit)
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for n in (4, 3, 2):
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            cand = toks[i : i + n]
            reason = _phrase_reject_reason(cand)
            if reason is None:
                continue
            phrase = " ".join(cand)
            key = (phrase, reason)
            if key in seen:
                continue
            seen.add(key)
            examples.append({"phrase": phrase, "reason": reason, "phrase_len": int(n)})
            if len(examples) >= max_examples:
                return examples
    return examples


def extract_evidence_phrases(text: Any, *, token_limit: int = 64) -> list[str]:
    toks = _phrase_tokens(text, token_limit=token_limit)
    phrases: list[str] = []
    seen: set[str] = set()
    for n in (4, 3, 2):
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            cand = toks[i : i + n]
            if not _valid_phrase_tokens(cand):
                continue
            phrase = " ".join(cand)
            if phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
    return phrases


def _phrase_score(
    phrase: str,
    *,
    count: int,
    doc_freq: int,
    total_docs: int,
    domain_role: str,
    entity: str,
) -> float:
    tokens = phrase.split()
    idf = math.log((float(total_docs) + 1.0) / (float(max(doc_freq, 1)) + 1.0)) + 1.0
    length_bonus = 0.25 * max(0, len(tokens) - 2)
    specificity_bonus = 0.0
    if any(tok in _CONCRETE_HINTS for tok in tokens):
        specificity_bonus += 1.1
    if any(tok in _NEGATIVE_KEEP_ADJECTIVES for tok in tokens):
        specificity_bonus += 0.35
    if entity == "item":
        specificity_bonus += 0.35
    if str(domain_role) == "target":
        specificity_bonus += 0.2
    repeated_bonus = 0.25 if int(count) >= 2 else 0.0
    generic_penalty = _phrase_generic_penalty(tokens)
    return round((math.log1p(max(1, int(count))) * idf) + length_bonus + specificity_bonus + repeated_bonus - generic_penalty, 6)


def _safe_int_text(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    raw = str(value).strip()
    if raw.endswith(".0"):
        try:
            return str(int(float(raw)))
        except Exception:
            return raw
    return raw


def _domain_role(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"auxiliary", "target"}:
        return raw
    return "target"


def _domain_dataset_pairs(auxiliary: str, target: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if str(auxiliary).strip():
        pairs.append(("auxiliary", str(auxiliary)))
    if str(target).strip():
        pairs.append(("target", str(target)))
    return pairs


def _read_train_history_frames(
    *,
    data_root: os.PathLike[str] | str,
    auxiliary: str,
    target: str,
) -> tuple[list[tuple[str, Path, pd.DataFrame]], dict[str, Any]]:
    frames: list[tuple[str, Path, pd.DataFrame]] = []
    sources: dict[str, Any] = {}
    for role, dataset in _domain_dataset_pairs(auxiliary, target):
        path = Path(data_root) / dataset / "train.csv"
        sources[f"{role}_train"] = file_fingerprint(path)
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0)
        cols = [
            c
            for c in (
                "user",
                "item",
                "user_idx",
                "item_idx",
                "user_idx_global",
                "item_idx_global",
                "explanation",
                "clean_text",
                "review",
            )
            if c in header.columns
        ]
        df = pd.read_csv(path, usecols=cols)
        frames.append((role, path, df))
    return frames, sources


def _text_from_history_row(row: Mapping[str, Any]) -> str:
    for key in ("clean_text", "explanation", "review"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _term_rows_from_counter(
    *,
    key: str,
    entity: str,
    domain_role: str,
    counter: Counter[str],
    doc_freq: Counter[str],
    total_docs: int,
    min_df: int,
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, int, int]] = []
    for phrase, count in counter.items():
        df = int(doc_freq.get(phrase, 0))
        if df < min_df:
            continue
        if not _valid_phrase_tokens(str(phrase).split()):
            continue
        score = _phrase_score(
            str(phrase),
            count=int(count),
            doc_freq=df,
            total_docs=int(total_docs),
            domain_role=domain_role,
            entity=entity,
        )
        candidates.append((score, str(phrase), int(count), df))
    candidates.sort(key=lambda item: (-item[0], -item[2], item[1]))
    for score, phrase, count, df in candidates[: max(top_n, 1)]:
        rows.append(
            {
                "key": key,
                "entity": entity,
                "domain_role": domain_role,
                "token": phrase,
                "phrase": phrase,
                "score": float(score),
                "count": int(count),
                "doc_freq": int(df),
                "phrase_len": int(len(phrase.split())),
            }
        )
    return rows


def build_history_term_tables(
    *,
    data_root: os.PathLike[str] | str,
    auxiliary: str,
    target: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    cfg = normalise_no_ref_evidence_config(config)
    frames, sources = _read_train_history_frames(data_root=data_root, auxiliary=auxiliary, target=target)
    user_counts: dict[str, Counter[str]] = defaultdict(Counter)
    item_counts: dict[str, Counter[str]] = defaultdict(Counter)
    key_domain: dict[str, str] = {}
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    doc_freq: Counter[str] = Counter()
    filter_reason_counts: Counter[str] = Counter()
    filter_reason_examples: list[dict[str, Any]] = []
    row_count = 0
    term_limit = int(cfg["term_token_limit"])
    for role, _path, df in frames:
        for row in df.to_dict("records"):
            history_text = _text_from_history_row(row)
            if len(filter_reason_examples) < 80:
                for item in collect_phrase_filter_audit(history_text, token_limit=term_limit, max_examples=12):
                    filter_reason_counts[str(item["reason"])] += 1
                    if len(filter_reason_examples) < 80:
                        sample = dict(item)
                        sample["domain_role"] = role
                        filter_reason_examples.append(sample)
            phrases = extract_evidence_phrases(history_text, token_limit=term_limit)
            if not phrases:
                continue
            row_count += 1
            unique = set(phrases)
            doc_freq.update(unique)
            phrase_counter = Counter(phrases)
            domain_counts[role].update(phrase_counter)
            user_keys = []
            item_keys = []
            for col in ("user_idx_global", "user_idx"):
                raw = _safe_int_text(row.get(col))
                if raw:
                    user_keys.append(f"{role}:idx:{raw}")
                    break
            raw_user = str(row.get("user", "") or "").strip()
            if raw_user:
                user_keys.append(f"{role}:raw:{raw_user}")
            for col in ("item_idx_global", "item_idx"):
                raw = _safe_int_text(row.get(col))
                if raw:
                    item_keys.append(f"{role}:idx:{raw}")
                    break
            raw_item = str(row.get("item", "") or "").strip()
            if raw_item:
                item_keys.append(f"{role}:raw:{raw_item}")
            for key in user_keys:
                user_counts[key].update(phrase_counter)
                key_domain[key] = role
            for key in item_keys:
                item_counts[key].update(phrase_counter)
                key_domain[key] = role
    min_df = int(cfg["min_df"])
    user_rows: list[dict[str, Any]] = []
    item_rows: list[dict[str, Any]] = []
    for key, counter in user_counts.items():
        user_rows.extend(
            _term_rows_from_counter(
                key=key,
                entity="user",
                domain_role=key_domain.get(key, "target"),
                counter=counter,
                doc_freq=doc_freq,
                total_docs=row_count,
                min_df=min_df,
                top_n=term_limit,
            )
        )
    for key, counter in item_counts.items():
        item_rows.extend(
            _term_rows_from_counter(
                key=key,
                entity="item",
                domain_role=key_domain.get(key, "target"),
                counter=counter,
                doc_freq=doc_freq,
                total_docs=row_count,
                min_df=min_df,
                top_n=term_limit,
            )
        )
    prior: dict[str, Any] = {
        "schema_version": "odcr_step5_no_ref_domain_prior/1",
        "min_df": min_df,
        "top_n": int(cfg["domain_prior_top_n"]),
        "domains": {},
    }
    for role, counter in domain_counts.items():
        scored_terms: list[tuple[float, str, int, int]] = []
        for phrase, count in counter.items():
            df = int(doc_freq.get(phrase, 0))
            if df < min_df:
                continue
            if not _valid_phrase_tokens(str(phrase).split()):
                continue
            score = _phrase_score(
                str(phrase),
                count=int(count),
                doc_freq=df,
                total_docs=int(row_count),
                domain_role=role,
                entity="domain",
            )
            scored_terms.append((score, str(phrase), int(count), df))
        scored_terms.sort(key=lambda item: (-item[0], -item[2], item[1]))
        terms = [
            {"token": phrase, "phrase": phrase, "score": float(score), "count": int(count), "doc_freq": int(df)}
            for score, phrase, count, df in scored_terms[: int(cfg["domain_prior_top_n"])]
        ]
        prior["domains"][role] = terms
    meta = {
        "history_row_count": int(row_count),
        "source_fingerprints": sources,
        "doc_freq_hash": stable_json_hash(dict(doc_freq.most_common(512))),
        "phrase_filter_audit": {
            "sampled_reason_counts": dict(filter_reason_counts.most_common(32)),
            "examples": filter_reason_examples[:80],
        },
    }
    term_columns = ["key", "entity", "domain_role", "token", "phrase", "score", "count", "doc_freq", "phrase_len"]
    return pd.DataFrame(user_rows, columns=term_columns), pd.DataFrame(item_rows, columns=term_columns), prior, meta


def _frame_identity(
    frame: pd.DataFrame,
    *,
    split_label: str,
    max_rows: int | None,
    include_label_hash: bool,
) -> dict[str, Any]:
    cols = [
        c
        for c in ("sample_id", "domain", "user_idx_global", "item_idx_global", "user_idx", "item_idx", "user", "item")
        if c in frame.columns
    ]
    if include_label_hash:
        cols.extend([c for c in ("clean_text", "explanation") if c in frame.columns])
    payload_df = frame[cols].copy() if cols else pd.DataFrame({"row_number": list(range(len(frame)))})
    hashed = pd.util.hash_pandas_object(payload_df.astype(str), index=False).values.tobytes()
    return {
        "split": str(split_label),
        "row_count": int(len(frame)),
        "max_rows": int(max_rows) if max_rows is not None and int(max_rows) > 0 else None,
        "identity_columns": cols,
        "row_hash": hashlib.sha256(hashed).hexdigest(),
    }


def _terms_mapping(term_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if term_df is None or len(term_df) == 0:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, group in term_df.groupby("key", sort=False):
        rows = []
        for r in group.to_dict("records"):
            phrase = str(r.get("phrase") or r.get("token") or "").strip().lower()
            if not phrase:
                continue
            rows.append(
                {
                    "phrase": phrase,
                    "score": float(r.get("score") or 0.0),
                    "count": int(r.get("count") or 0),
                    "doc_freq": int(r.get("doc_freq") or 0),
                    "source_key": str(key),
                    "entity": str(r.get("entity") or ""),
                    "domain_role": str(r.get("domain_role") or ""),
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), -int(item["count"]), str(item["phrase"])))
        out[str(key)] = rows
    return out


def _candidate_keys(row: Mapping[str, Any], *, entity: str) -> list[str]:
    role = _domain_role(row.get("domain"))
    out: list[str] = []
    if entity == "user":
        for col in ("user_idx_global", "user_idx"):
            raw = _safe_int_text(row.get(col))
            if raw:
                out.append(f"{role}:idx:{raw}")
        raw_user = str(row.get("user", "") or "").strip()
        if raw_user:
            out.append(f"{role}:raw:{raw_user}")
    else:
        for col in ("item_idx_global", "item_idx"):
            raw = _safe_int_text(row.get(col))
            if raw:
                out.append(f"{role}:idx:{raw}")
        raw_item = str(row.get("item", "") or "").strip()
        if raw_item:
            out.append(f"{role}:raw:{raw_item}")
    return out


def _lookup_phrase_records(
    row: Mapping[str, Any],
    *,
    entity: str,
    mapping: Mapping[str, Sequence[Mapping[str, Any]]],
    banned: set[str],
    top_k: int,
    route_weighted: bool = False,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    route_bonus = 0.0
    target_gold_bonus = 0.0
    if route_weighted:
        try:
            route_bonus += 0.35 if int(float(row.get("route_explainer", 0) or 0)) == 1 else 0.0
        except Exception:
            pass
        try:
            route_bonus += 0.15 * max(0.0, float(row.get("confidence_bucket", 0) or 0.0))
        except Exception:
            pass
        try:
            route_bonus += 0.10 * max(0.0, float(row.get("sample_weight_hint", 0) or 0.0))
        except Exception:
            pass
        origin = str(row.get("sample_origin") or "").strip().lower()
        if origin == "target_gold":
            target_gold_bonus = 0.35
        elif origin == "aux_gold":
            target_gold_bonus = 0.05
        route_bonus += target_gold_bonus
    for key in _candidate_keys(row, entity=entity):
        for raw in mapping.get(key, []):
            phrase = str(raw.get("phrase") or "").strip().lower()
            tokens = phrase.split()
            if not phrase or phrase in seen:
                continue
            if any(tok in banned for tok in tokens):
                continue
            if not _valid_phrase_tokens(tokens):
                continue
            seen.add(phrase)
            item = dict(raw)
            item["phrase"] = phrase
            item["base_score"] = float(raw.get("score") or 0.0)
            item["score"] = round(float(raw.get("score") or 0.0) + route_bonus, 6)
            item["route_confidence_bonus"] = round(route_bonus, 6)
            item["target_gold_bonus"] = round(target_gold_bonus, 6)
            candidates.append(item)
    candidates.sort(key=lambda item: (-float(item["score"]), -int(item.get("count") or 0), str(item["phrase"])))
    return candidates[: max(0, int(top_k))]


def _domain_prior_terms(
    row: Mapping[str, Any],
    *,
    domain_prior: Mapping[str, Any],
    banned: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    role = _domain_role(row.get("domain"))
    terms = ((domain_prior.get("domains") or {}).get(role) or [])
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in terms:
        token = str((raw or {}).get("phrase") or (raw or {}).get("token") or "").strip().lower()
        tokens = token.split()
        if not token or token in seen:
            continue
        if any(tok in banned for tok in tokens):
            continue
        if not _valid_phrase_tokens(tokens):
            continue
        seen.add(token)
        out.append(
            {
                "phrase": token,
                "score": float((raw or {}).get("score") or 0.0),
                "count": int((raw or {}).get("count") or 0),
                "doc_freq": int((raw or {}).get("doc_freq") or 0),
                "source_key": f"domain:{role}",
                "entity": "domain",
                "domain_role": role,
            }
        )
        if len(out) >= top_k:
            break
    return out


def _phrase_text(raw: Any) -> str:
    if isinstance(raw, Mapping):
        return str(raw.get("phrase") or raw.get("token") or "").strip().lower()
    return str(raw or "").strip().lower()


def _compact_phrase_terms(terms: Sequence[Any], *, top_k: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        tok = _phrase_text(raw)
        if not tok or tok in seen:
            continue
        if not _valid_phrase_tokens(tok.split()):
            continue
        seen.add(tok)
        out.append(tok.replace("_", " "))
        if len(out) >= max(0, int(top_k)):
            break
    return out


def _join_phrases(terms: Sequence[Any], *, empty: str = "none") -> str:
    cleaned = _compact_phrase_terms(terms, top_k=len(terms))
    return "; ".join(cleaned) if cleaned else empty


def _build_compact_prompt(
    *,
    row: Mapping[str, Any],
    protocol: str,
    cfg: Mapping[str, Any],
    item_top: Sequence[str],
    user_top: Sequence[str],
    domain_top: Sequence[str],
    negative_top: Sequence[str],
) -> str:
    if protocol == SOFT_PREFIX_LIGHT_NO_REF:
        required = (
            "route_scorer",
            "route_explainer",
            "confidence_bucket",
            "sample_weight_hint",
            "sample_origin",
        )
        missing = [key for key in required if key not in row or str(row.get(key, "")).strip() == ""]
        raise ValueError(
            "soft_prefix_light_no_ref requires resolved continuous Step3/Step4 latent metadata before use; "
            f"missing={missing or ['continuous_prefix_tensor']}"
        )
    if protocol in RETIRED_INPUT_PROTOCOLS:
        raise ValueError(f"retired Step5 no-ref protocol cannot build prompts: {protocol!r}")
    task = str(cfg.get("concise_prompt") or "Write one concise review reason using the evidence.").strip()
    item_text = _join_phrases(item_top)
    user_text = _join_phrases(user_top)
    domain_text = _join_phrases(domain_top)
    if protocol == TEXT_CLEAN_ITEM_ONLY_NO_REF:
        parts = [f"Item evidence: {item_text}."]
        if item_text == "none" and domain_text != "none":
            parts.append(f"Domain prior: {domain_text}.")
        parts.append(f"Task: {task}")
        return "\n".join(parts)
    if protocol == TEXT_CLEAN_ITEM_USER_NO_REF:
        parts = [f"Item evidence: {item_text}."]
        if user_text != "none":
            parts.append(f"User preference: {user_text}.")
        if item_text == "none" and user_text == "none" and domain_text != "none":
            parts.append(f"Domain prior: {domain_text}.")
        parts.append(f"Task: {task}")
        return "\n".join(parts)
    if protocol == ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF:
        parts = [f"Item evidence: {item_text}."]
        if item_text == "none" and domain_text != "none":
            parts.append(f"Domain prior: {domain_text}.")
        parts.append(f"Task: {task}")
        return "\n".join(parts)
    if protocol == NEUTRAL_CORE_NO_REF:
        return f"Item evidence: none.\nTask: {task}"
    raise ValueError(f"unsupported Step5 no-ref protocol: {protocol!r}")


def build_no_ref_evidence_rows(
    frame: pd.DataFrame,
    *,
    split_label: str,
    config: Mapping[str, Any],
    user_terms: pd.DataFrame,
    item_terms: pd.DataFrame,
    domain_prior: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    cfg = normalise_no_ref_evidence_config(config)
    protocol = str(cfg["input_protocol"])
    neutral = str(cfg["neutral_content_evidence"])
    user_map = _terms_mapping(user_terms)
    item_map = _terms_mapping(item_terms)
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    split_norm = str(split_label).strip().lower()
    for pos, row in enumerate(frame.to_dict("records")):
        current_label_tokens = set()
        if split_norm == "train":
            current_label_tokens = set(_normalised_tokens(row.get("clean_text", row.get("explanation", ""))))
        if protocol == NEUTRAL_CORE_NO_REF:
            text = _build_compact_prompt(
                row=row,
                protocol=protocol,
                cfg=cfg,
                item_top=[],
                user_top=[],
                domain_top=[],
                negative_top=[],
            )
            texts.append(text)
            metas.append(
                {
                    "row_number": int(pos),
                    "input_protocol": protocol,
                    "content_evidence_source": "neutral_core_no_ref",
                    "history_available": False,
                    "user_history_terms": 0,
                    "item_history_terms": 0,
                    "domain_prior_terms": 0,
                }
            )
            continue
        route_weighted = protocol == ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF
        user_records = _lookup_phrase_records(
            row,
            entity="user",
            mapping=user_map,
            banned=current_label_tokens,
            top_k=int(cfg["user_history_top_k"]),
            route_weighted=False,
        )
        item_records = _lookup_phrase_records(
            row,
            entity="item",
            mapping=item_map,
            banned=current_label_tokens,
            top_k=int(cfg["item_history_top_k"]),
            route_weighted=route_weighted,
        )
        domain_records = _domain_prior_terms(
            row,
            domain_prior=domain_prior,
            banned=current_label_tokens,
            top_k=int(cfg["domain_prior_display_top_k"]),
        )
        user_top = [r["phrase"] for r in user_records]
        item_top = [r["phrase"] for r in item_records]
        domain_top = [r["phrase"] for r in domain_records]
        used_positive = set(item_top) | set(user_top)
        negative_top = [tok for tok in domain_top if tok not in used_positive]
        if not user_top and not item_top and not domain_top:
            text = _build_compact_prompt(
                row=row,
                protocol=protocol,
                cfg=cfg,
                item_top=[],
                user_top=[],
                domain_top=[],
                negative_top=[],
            )
            source = "neutral_no_train_history"
        else:
            text = _build_compact_prompt(
                row=row,
                protocol=protocol,
                cfg=cfg,
                item_top=item_top,
                user_top=user_top,
                domain_top=domain_top,
                negative_top=negative_top,
            )
            if protocol == ROUTE_WEIGHTED_ITEM_PHRASE_V2_NO_REF:
                source = "route_weighted_item_phrase_v3_history"
            elif protocol == TEXT_CLEAN_ITEM_USER_NO_REF:
                source = "item_user_phrase_v3_train_history"
            else:
                source = "item_phrase_v3_train_history"
            if not user_top and not item_top:
                source = "phrase_v3_domain_prior"
        texts.append(text)
        metas.append(
            {
                "row_number": int(pos),
                "input_protocol": protocol,
                "content_evidence_source": source,
                "history_available": bool(user_top or item_top or domain_top),
                "user_history_terms": int(len(user_top)),
                "item_history_terms": int(len(item_top)),
                "domain_prior_terms": int(len(domain_top)),
                "encoder_content_token_budget": int(cfg["encoder_content_token_budget"]),
                "item_evidence_phrases": item_records,
                "user_preference_phrases": user_records,
                "domain_prior_phrases": domain_records,
                "route_weighted": bool(route_weighted),
            }
        )
    return texts, metas


def build_or_load_no_ref_evidence_for_frame(
    frame: pd.DataFrame,
    *,
    split_label: str,
    task_id: int,
    auxiliary: str,
    target: str,
    config: Mapping[str, Any],
    data_root: os.PathLike[str] | str,
    cache_root: os.PathLike[str] | str,
    max_rows: int | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    cfg = normalise_no_ref_evidence_config(config)
    split_norm = str(split_label).strip().lower()
    role_file = {
        "train": "train_evidence_selected.parquet",
        "valid": "valid_evidence.parquet",
        "test": "test_evidence.parquet",
    }.get(split_norm, f"{split_norm}_evidence.parquet")
    frame_identity = _frame_identity(
        frame,
        split_label=split_norm,
        max_rows=max_rows if bool(cfg.get("smoke_cache_identity_includes_max_rows", True)) else None,
        include_label_hash=(split_norm == "train"),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "task_id": int(task_id),
        "auxiliary": str(auxiliary),
        "target": str(target),
        "config": cfg,
        "frame_identity": frame_identity,
    }
    cache_hash = stable_json_hash(identity, length=32)
    root = Path(cache_root) / f"task{int(task_id)}" / cache_hash
    manifest_path = root / "manifest.json"
    evidence_path = root / role_file
    user_terms_path = root / "user_terms.parquet"
    item_terms_path = root / "item_terms.parquet"
    domain_prior_path = root / "domain_prior.json"
    if evidence_path.exists() and manifest_path.exists() and user_terms_path.exists() and item_terms_path.exists() and domain_prior_path.exists():
        ev_df = pd.read_parquet(evidence_path)
        texts = ev_df["content_evidence"].fillna("").astype(str).tolist()
        metas = [
            {
                "content_evidence_source": str(row.get("content_evidence_source") or ""),
                "input_protocol": str(row.get("input_protocol") or cfg["input_protocol"]),
                "history_available": bool(row.get("history_available", False)),
                "user_history_terms": int(row.get("user_history_terms", 0) or 0),
                "item_history_terms": int(row.get("item_history_terms", 0) or 0),
                "domain_prior_terms": int(row.get("domain_prior_terms", 0) or 0),
                "item_evidence_phrases": json.loads(str(row.get("item_evidence_phrases_json") or "[]")),
                "user_preference_phrases": json.loads(str(row.get("user_preference_phrases_json") or "[]")),
                "domain_prior_phrases": json.loads(str(row.get("domain_prior_phrases_json") or "[]")),
                "cache_hit": True,
            }
            for row in ev_df.to_dict("records")
        ]
        return texts, metas, json.loads(manifest_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    term_columns = ["key", "entity", "domain_role", "token", "phrase", "score", "count", "doc_freq", "phrase_len"]
    if cfg["input_protocol"] == NEUTRAL_CORE_NO_REF:
        user_terms = pd.DataFrame([], columns=term_columns)
        item_terms = pd.DataFrame([], columns=term_columns)
        domain_prior = {
            "schema_version": "odcr_step5_no_ref_domain_prior/1",
            "min_df": int(cfg["min_df"]),
            "top_n": int(cfg["domain_prior_top_n"]),
            "domains": {"auxiliary": [], "target": []},
        }
        history_meta = {
            "history_row_count": 0,
            "source_fingerprints": {
                f"{role}_train": file_fingerprint(Path(data_root) / dataset / "train.csv")
                for role, dataset in _domain_dataset_pairs(auxiliary, target)
            },
            "doc_freq_hash": stable_json_hash({}),
            "neutral_core_skipped_history_scan": True,
        }
    elif cfg["input_protocol"] == SOFT_PREFIX_LIGHT_NO_REF:
        raise ValueError(
            "soft_prefix_light_no_ref requires resolved continuous prefix metadata; "
            "it is retired-fail-fast until the Step3/Step4 soft-prefix tensor contract is available."
        )
    else:
        user_terms, item_terms, domain_prior, history_meta = build_history_term_tables(
            data_root=data_root,
            auxiliary=auxiliary,
            target=target,
            config=cfg,
        )
    user_terms.to_parquet(user_terms_path, index=False)
    item_terms.to_parquet(item_terms_path, index=False)
    domain_prior_path.write_text(json.dumps(domain_prior, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    texts, metas = build_no_ref_evidence_rows(
        frame,
        split_label=split_norm,
        config=cfg,
        user_terms=user_terms,
        item_terms=item_terms,
        domain_prior=domain_prior,
    )
    ev_df = pd.DataFrame(
        {
            "row_number": list(range(len(texts))),
            "sample_id": frame["sample_id"].tolist() if "sample_id" in frame.columns else list(range(len(texts))),
            "domain": frame["domain"].astype(str).tolist() if "domain" in frame.columns else ["target"] * len(texts),
            "content_evidence": texts,
            "content_evidence_source": [m["content_evidence_source"] for m in metas],
            "input_protocol": [m["input_protocol"] for m in metas],
            "history_available": [bool(m["history_available"]) for m in metas],
            "user_history_terms": [int(m["user_history_terms"]) for m in metas],
            "item_history_terms": [int(m["item_history_terms"]) for m in metas],
            "domain_prior_terms": [int(m["domain_prior_terms"]) for m in metas],
            "item_evidence_phrases_json": [
                json.dumps(m.get("item_evidence_phrases") or [], ensure_ascii=True, sort_keys=True) for m in metas
            ],
            "user_preference_phrases_json": [
                json.dumps(m.get("user_preference_phrases") or [], ensure_ascii=True, sort_keys=True) for m in metas
            ],
            "domain_prior_phrases_json": [
                json.dumps(m.get("domain_prior_phrases") or [], ensure_ascii=True, sort_keys=True) for m in metas
            ],
        }
    )
    for col in ("user_idx_global", "item_idx_global"):
        if col in frame.columns:
            ev_df[col] = frame[col].tolist()
    ev_df.to_parquet(evidence_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "cache_hash": cache_hash,
        "task_id": int(task_id),
        "auxiliary": str(auxiliary),
        "target": str(target),
        "input_protocol": cfg["input_protocol"],
        "split": split_norm,
        "role_file": role_file,
        "row_count": int(len(texts)),
        "max_rows": frame_identity.get("max_rows"),
        "identity": identity,
        "history": history_meta,
        "files": {
            "user_terms": str(user_terms_path),
            "item_terms": str(item_terms_path),
            "domain_prior": str(domain_prior_path),
            role_file: str(evidence_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    for meta in metas:
        meta["cache_hit"] = False
    return texts, metas, manifest
