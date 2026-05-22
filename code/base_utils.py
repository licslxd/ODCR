import os

# 必须在首次 import nltk 之前绑定本地数据，否则 METEOR / word_tokenize 可能触发 nltk.download 联网
from paths_config import (
    get_meteor_cache_dir,
    get_meteor_metric_module_dir,
    require_nltk_data_dir,
)

_NLTK_LOCAL = require_nltk_data_dir()
os.environ["NLTK_DATA"] = _NLTK_LOCAL

import re
from typing import Any, Dict, List, Sequence, Tuple

from rouge import rouge, rouge_from_word_lists
from nltk import word_tokenize
from bleu import compute_bleu
import logging
import nltk

_logger = logging.getLogger(__name__)

if _NLTK_LOCAL not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_LOCAL)


def _patch_nltk_download_offline_only():
    """HF evaluate 的 METEOR 在 _download_and_prepare 里无条件调用 nltk.download，离线仍会联网。

    若本地已有对应语料（与 meteor.py 中 wordnet / punkt / punkt_tab / omw-1.4 一致），则直接返回成功，
    避免 [nltk_data] urlopen / Name or service not known。
    """
    _FIND = {
        "wordnet": "corpora/wordnet",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "omw-1.4": "corpora/omw-1.4",
    }
    if getattr(nltk, "_odcr_offline_dl_patch", False):
        return
    _orig = nltk.download

    def _wrapped(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0] in _FIND:
            try:
                nltk.data.find(_FIND[args[0]])
                return True
            except LookupError:
                return False
        return _orig(*args, **kwargs)

    nltk.download = _wrapped
    nltk._odcr_offline_dl_patch = True


from torch import nn
import torch
import math


def get_underlying_model(model):
    """当 model 被 DistributedDataParallel 包装时返回原始模块。"""
    if isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module
    return model

def T5_shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
        " See T5 docs for more information"
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def compute_bleu1234_only(predictions, references):
    """
    仅计算 BLEU-1~4（与 evaluate_text 中词级 BLEU 口径一致），用于训练阶段按验证集 BLEU-4 选模，避免 METEOR 等额外评测开销。
    """
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    out = {}
    for order in (1, 2, 3, 4):
        try:
            bleu_n, _, _, _, _, _ = compute_bleu(
                formatted_ref, predictions_tokens, max_order=order, smooth=False
            )
            out[str(order)] = round(bleu_n * 100, 2)
        except Exception:
            out[str(order)] = 0.0
    return out


def evaluate_text(predictions, references):
    """
    Example:
        >>> predictions = ["good day", "need to work"]
        >>> references = ["nice day", "work from home"]
        >>> evlauate_text(predictions, references)
    """
    # compute bleu
    # compute rouge
    # compute distinct（语料级 n-gram distinct，×100；论文主表 DIST-1/DIST-2 口径；
    #   与 odcr_eval_metrics.extended_text_metrics_bundle 中的 distinct 定义不同，勿混读）
    # compute meteor

    def distinct_score(sentences, n):
        sentences = [word_tokenize(sentence) for sentence in sentences]
        unique_ngrams = set()
        total_ngrams = 0

        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
            unique_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        distinct_score = len(unique_ngrams) / total_ngrams
        return distinct_score
    # dist score
    try:
        dist1 = round(distinct_score(predictions, 1) * 100, 2)
    except:
        dist1 = 0
    try:
        dist2 = round(distinct_score(predictions, 2) * 100, 2)
    except:
        dist2 = 0
    
    # bleu score
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    try:
        bleu1, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=1, smooth=False)
        bleu1 = round(bleu1*100, 2)
    except:
        bleu1 = 0
    try:
        bleu2, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=2, smooth=False)
        bleu2 = round(bleu2*100, 2)
    except:
        bleu2 = 0
    try:
        bleu3, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=3, smooth=False)
        bleu3 = round(bleu3*100, 2)
    except:
        bleu3 = 0
    try:
        bleu4, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=4, smooth=False)
        bleu4 = round(bleu4*100,2)
    except:
        bleu4 = 0
    
    # rouge score
    score = rouge(predictions, references)
    rouge_s = {k: round(v * 100, 2) for (k, v) in score.items()}
    
    
    # meteor score (离线：使用本地 cache_dir，无缓存时跳过)
    try:
        import evaluate
        _cache = get_meteor_cache_dir()
        os.makedirs(_cache, exist_ok=True)
        # 必须早于 evaluate.load：meteor 指标会无条件 nltk.download，需改为仅检测本地
        _patch_nltk_download_offline_only()
        # 优先从 hf_cache 中的本地 meteor.py 加载，避免 evaluate.load("meteor") 在离线时打印 Hub 解析提示
        _meteor_dir = get_meteor_metric_module_dir()
        _meteor_script = None
        if os.path.isdir(_meteor_dir):
            for _entry in sorted(os.listdir(_meteor_dir)):
                _candidate = os.path.join(_meteor_dir, _entry, "meteor.py")
                if os.path.isfile(_candidate):
                    _meteor_script = _candidate
                    break
        if _meteor_script:
            meteor = evaluate.load(_meteor_script, cache_dir=_cache)
        else:
            meteor = evaluate.load("meteor", cache_dir=_cache)
        meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
        meteor_score = round(meteor_score * 100, 2)
    except Exception:
        _logger.exception("METEOR failed")
        meteor_score = 0.0

    return {
        "rouge": {"1": rouge_s["rouge_1/f_score"], "2": rouge_s["rouge_2/f_score"], "l": rouge_s["rouge_l/f_score"]},
        "bleu": {"1": bleu1, "2": bleu2, "3": bleu3, "4": bleu4},
        "dist": {"1": dist1, "2": dist2},
        "meteor": meteor_score,
    }


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]`` (batch_first)
        """
        if x.dim() == 3:
            # batch_first: (N, L, D) -> add pe (1, L, D)
            seq_len = x.size(1)
            x = x + self.pe[:seq_len].transpose(0, 1)
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_count_mask(tgt_len, device):
    src_len = 3
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[1, 2] = False
    return mask

def generate_new_mask(tgt_len, device):
    src_len = 3
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[1, 2] = False
    return mask

def generate_domain_mask(tgt_len, device):
    src_len = 5  # Set src_len to 5
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[:src_len, :src_len] = False  # No masking for the first 5 positions
    return mask

def generate_peter_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    return mask

def generate_square_mask(seqlen, device):
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device), diagonal=1) == 1
    return mask

def generate_peter_noui_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[2:,:2] = True
    return mask

def compute_entropy(generated_dist):
    log_probabilities = torch.log(generated_dist + 1e-9)  # Ensure no log(0) by adding a small epsilon
    entropies = -torch.sum(generated_dist * log_probabilities, dim=-1)  # Shape (N, seqlen)
    sample_entropies = torch.mean(entropies, dim=-1)  # Shape (N,)
    return sample_entropies

def filter_by_entropy(entropy_values, percentile=0.75):
    entropy_tensor = torch.tensor(entropy_values)
    threshold = torch.quantile(entropy_tensor, percentile)
    filtered_indices = torch.where(entropy_tensor <= threshold)[0]
    return filtered_indices.tolist()


PAPER_METRICS_SCHEMA_VERSION = "odcr_paper_comparable_text/1.0"
OFFICIAL_PAPER_METRICS_SCHEMA_VERSION = "odcr_step5_official_paper_metrics/1"
PAPER_METRIC_INPUT_SCHEMA_VERSION = "odcr_step5_paper_metric_inputs/1"


def _tokenizer_truncate_decode_text(text: Any, tokenizer: Any, *, max_len: int) -> Tuple[str, List[int], int]:
    if tokenizer is None:
        raise ValueError("official paper metric input builder requires tokenizer")
    if int(max_len) <= 0:
        raise ValueError(f"official paper metric max_len must be positive, got {max_len!r}")
    raw = "" if text is None else str(text)
    encoded = tokenizer(raw, add_special_tokens=False, truncation=False)
    original_ids = [int(x) for x in list(encoded.get("input_ids") or [])]
    ids = original_ids[: int(max_len)]
    try:
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
    except TypeError:
        decoded = tokenizer.decode(ids)
    return str(decoded).strip(), ids, int(len(original_ids))


def build_paper_metric_inputs(pred_text: Any, ref_text: Any, tokenizer: Any, max_len: int = 25) -> Dict[str, Any]:
    """Build the only official Step5_e paper metric input pair.

    Both sides are re-tokenized from raw text, truncated to the same token
    budget, then decoded before metric computation. Official metrics must use
    ``metric_pred`` and ``metric_ref`` from this function, not raw pred/ref text.
    """
    metric_pred, pred_ids, pred_original_len = _tokenizer_truncate_decode_text(pred_text, tokenizer, max_len=max_len)
    metric_ref, ref_ids, ref_original_len = _tokenizer_truncate_decode_text(ref_text, tokenizer, max_len=max_len)
    return {
        "schema_version": PAPER_METRIC_INPUT_SCHEMA_VERSION,
        "max_len": int(max_len),
        "metric_pred": metric_pred,
        "metric_ref": metric_ref,
        "prediction_token_count": int(len(pred_ids)),
        "reference_token_count": int(len(ref_ids)),
        "prediction_original_token_count": int(pred_original_len),
        "reference_original_token_count": int(ref_original_len),
        "prediction_truncated": bool(pred_original_len > int(max_len)),
        "reference_truncated": bool(ref_original_len > int(max_len)),
    }


def paper_tokenize_words(s: str) -> List[str]:
    """论文可比指标统一分词：NLTK word_tokenize，失败时回退非空白切分。"""
    try:
        return word_tokenize(s or "")
    except Exception:
        return re.findall(r"\S+", s or "")


def compute_paper_comparable_text_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    """
    BLEU / ROUGE / 语料级 distinct 均基于 ``paper_tokenize_words``，ROUGE 与 BLEU 词边界一致。
    显式区分 ratio_0_1（原始比）与 percent_0_100（百分制）。
    """
    pred_list = [str(p) if p is not None else "" for p in predictions]
    ref_list = [str(r) if r is not None else "" for r in references]
    ptoks = [paper_tokenize_words(p) for p in pred_list]
    rtoks = [paper_tokenize_words(r) for r in ref_list]
    formatted_ref = [[r] for r in rtoks]

    bleu_pct: Dict[str, float] = {}
    bleu_raw: Dict[str, float] = {}
    for order in (1, 2, 3, 4):
        try:
            bleu_n, _, _, _, _, _ = compute_bleu(formatted_ref, ptoks, max_order=order, smooth=False)
            bleu_raw[str(order)] = round(float(bleu_n), 6)
            bleu_pct[str(order)] = round(float(bleu_n) * 100.0, 4)
        except Exception:
            bleu_raw[str(order)] = 0.0
            bleu_pct[str(order)] = 0.0

    rscores = rouge_from_word_lists(ptoks, rtoks)
    rouge_ratio = {k: round(float(v), 6) for k, v in rscores.items()}
    rouge_pct = {k: round(float(v) * 100.0, 4) for k, v in rscores.items()}

    def _corpus_dist_ratio(sent_toks: List[List[str]], n: int) -> float:
        unique = set()
        total = 0
        for toks in sent_toks:
            if n == 1:
                grams = [(t,) for t in toks]
            else:
                grams = [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]
            unique.update(grams)
            total += len(grams)
        if total <= 0:
            return 0.0
        return float(len(unique)) / float(total)

    d1r = _corpus_dist_ratio(ptoks, 1)
    d2r = _corpus_dist_ratio(ptoks, 2)

    return {
        "schema_version": PAPER_METRICS_SCHEMA_VERSION,
        "tokenization": {
            "name": "nltk_word_tokenize_with_regex_fallback",
            "module": "base_utils.paper_tokenize_words",
        },
        "bleu": {
            "scale": "percent_0_100",
            "1": bleu_pct["1"],
            "2": bleu_pct["2"],
            "3": bleu_pct["3"],
            "4": bleu_pct["4"],
        },
        "bleu_raw_ratio": {
            "scale": "ratio_0_1",
            "1": bleu_raw["1"],
            "2": bleu_raw["2"],
            "3": bleu_raw["3"],
            "4": bleu_raw["4"],
        },
        "rouge": {
            "scale": "percent_0_100",
            "rouge_1_f": rouge_pct["rouge_1/f_score"],
            "rouge_2_f": rouge_pct["rouge_2/f_score"],
            "rouge_l_f": rouge_pct["rouge_l/f_score"],
        },
        "rouge_raw_ratio": {"scale": "ratio_0_1", **rouge_ratio},
        "distinct_corpus": {
            "scale_percent_0_100": {"1": round(d1r * 100.0, 4), "2": round(d2r * 100.0, 4)},
            "scale_ratio_0_1": {"1": round(d1r, 6), "2": round(d2r, 6)},
        },
        "note": "repo evaluate_text 仍保留（含空格切分 ROUGE）；本块为统一分词后的论文对照口径",
    }


def official_paper_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    """Single official Step5_e paper metric implementation.

    Inputs must already be the 25-token ``metric_pred`` / ``metric_ref`` texts
    produced by :func:`build_paper_metric_inputs`.
    """
    pred_list = [str(p) if p is not None else "" for p in predictions]
    ref_list = [str(r) if r is not None else "" for r in references]
    if len(pred_list) != len(ref_list):
        raise ValueError(f"official paper metrics length mismatch: {len(pred_list)} != {len(ref_list)}")
    paper = compute_paper_comparable_text_metrics(pred_list, ref_list)
    try:
        import evaluate
        _cache = get_meteor_cache_dir()
        os.makedirs(_cache, exist_ok=True)
        _patch_nltk_download_offline_only()
        _meteor_dir = get_meteor_metric_module_dir()
        _meteor_script = None
        if os.path.isdir(_meteor_dir):
            for _entry in sorted(os.listdir(_meteor_dir)):
                _candidate = os.path.join(_meteor_dir, _entry, "meteor.py")
                if os.path.isfile(_candidate):
                    _meteor_script = _candidate
                    break
        if _meteor_script:
            meteor = evaluate.load(_meteor_script, cache_dir=_cache)
        else:
            meteor = evaluate.load("meteor", cache_dir=_cache)
        meteor_score = round(float(meteor.compute(predictions=pred_list, references=ref_list)["meteor"]) * 100.0, 4)
    except Exception:
        _logger.exception("official METEOR failed")
        meteor_score = 0.0
    return {
        "schema_version": OFFICIAL_PAPER_METRICS_SCHEMA_VERSION,
        "input_schema_version": PAPER_METRIC_INPUT_SCHEMA_VERSION,
        "token_length_policy": {
            "prediction_max_length": 25,
            "reference_max_length": 25,
            "unit": "tokenizer_tokens_then_decode",
        },
        "rouge": paper["rouge"],
        "bleu": paper["bleu"],
        "meteor": meteor_score,
        "distinct_corpus": paper["distinct_corpus"],
        "tokenization": paper["tokenization"],
        "scale": "percent_0_100",
    }


def diagnostic_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Dict[str, Any]:
    """Non-official text diagnostics; never use for Step5 best selection."""
    return {
        "schema_version": "odcr_step5_diagnostic_metrics/1",
        "official_paper_metrics": False,
        "evaluate_text": evaluate_text(predictions, references),
    }
