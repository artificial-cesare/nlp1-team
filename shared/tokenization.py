"""Shared preprocessing and tokenisation for NLP1 Project 1 (Name That LLM).

    from shared.tokenization import preprocess, tokenize, tokenizer_stats, TOKENIZERS
    ids = tokenize(texts, "roberta-bpe")          # list of token-id lists
    tokenizer_stats(texts)                        # fertility, truncation@512, unk rate, typography round trip

Preprocessing is NFC only (no lowercasing, no stopword or punctuation removal):
punctuation and typography are authorship cues. See part1/neural_project.ipynb, steps 3-4.
"""

from __future__ import annotations

import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"

TOKENIZERS = {  # name -> Hugging Face id, "tutorial", or "ours-bpe:<vocab size>" (local BPEs, built on first use)
    "roberta-bpe": "rrivera1849/LUSR",          # LUSR; provisional vocabulary for the from-scratch transformer
    "qwen3-bpe": "Qwen/Qwen3-Embedding-0.6B",   # identical to Qwen/Qwen3-0.6B-Base's tokeniser
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta-v3": "microsoft/deberta-v3-base",
    "tutorial-bpe-1k": "tutorial",              # part1.experiment.train_tokenizer on AG News, vocab 1000, min_freq 2
    # Kazik's BPE (part1/tok.py): course BPE trained on the train split, punctuation isolated, min_freq 2.
    "ours-bpe-2k": "ours-bpe:2000",
    "ours-bpe-4k": "ours-bpe:4000",
    "ours-bpe-8k": "ours-bpe:8000",             # closest to the pretrained tokenisers' tokens per word (comparison)
}

# The tokenisers our models actually use (Kazik: evaluate these). roberta-bpe = LUSR (frozen and fine-tuned) and the
# from-scratch transformer's vocabulary; qwen3-bpe = Qwen3-0.6B-Base hidden states (and the dropped Qwen3-Embedding).
# modernbert / deberta-v3 / tutorial-bpe-1k are comparison rows only (step 4).
USED_TOKENIZERS = ["roberta-bpe", "qwen3-bpe"]

# Kazik's tokeniser evaluation (part1/project.ipynb, step 2) hands these over for the downstream models: 2k and 4k
# chosen on CV folds, 8k as the comparison closest to the pretrained tokenisers. Downstream accuracy decides.
OURS_TOKENIZERS = ["ours-bpe-2k", "ours-bpe-4k", "ours-bpe-8k"]

PROBE = "He said “no” — it’s 5 µm…  done"  # curly quotes, em dash, µ, ellipsis, double space


def preprocess(text: str) -> str:
    """NFC only. NFKC would rewrite superscripts, '…' and non-breaking spaces, erasing cues."""
    return unicodedata.normalize("NFC", text)


def _is_local(name: str) -> bool:
    """A locally trained `tokenizers` BPE (tutorial or ours), not a Hugging Face tokeniser."""
    return TOKENIZERS[name] == "tutorial" or TOKENIZERS[name].startswith("ours-bpe:")


@lru_cache(maxsize=None)
def get_tokenizer(name: str):
    if not _is_local(name):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZERS[name])
    if TOKENIZERS[name].startswith("ours-bpe:"):  # trained on the train split on first use, cached in cache/
        sys.path.insert(0, str(ROOT))
        from part1.tok import final_bpe

        return final_bpe(int(TOKENIZERS[name].split(":")[1]))
    from tokenizers import Tokenizer

    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / "tutorial_bpe_agnews_vs1000_mf2.json"
    if not path.exists():
        sys.path.insert(0, str(ROOT))
        import part1.experiment

        corpus = part1.experiment.download_ag_news_texts()
        part1.experiment.train_tokenizer(corpus, vocab_size=1000, min_frequency=2).save(str(path))
    return Tokenizer.from_file(str(path))


def tokenize(texts, tokenizer_name: str) -> list[list[int]]:
    """Token ids per text (no special tokens, no truncation), after NFC preprocessing."""
    tok, texts = get_tokenizer(tokenizer_name), [preprocess(t) for t in texts]
    if _is_local(tokenizer_name):
        return [e.ids for e in tok.encode_batch(texts)]
    return tok(texts, add_special_tokens=False)["input_ids"]


def unk_id(tokenizer_name: str):
    tok = get_tokenizer(tokenizer_name)
    return tok.token_to_id("[UNK]") if _is_local(tokenizer_name) else tok.unk_token_id


def roundtrip(text: str, tokenizer_name: str) -> str:
    """Text as the tokeniser sees it. For local BPEs we join tokens, since their decode drops [UNK]."""
    tok = get_tokenizer(tokenizer_name)
    if _is_local(tokenizer_name):
        return "".join(tok.encode(preprocess(text)).tokens).replace("▁", " ").strip()
    return tok.decode(tokenize([text], tokenizer_name)[0])


def tokenizer_stats(texts, names=tuple(TOKENIZERS), max_len: int = 512) -> pd.DataFrame:
    """Fertility (tokens / whitespace word), share truncated at max_len (incl. 2 special tokens), unk rate."""
    texts = list(texts)
    n_words = sum(len(t.split()) for t in texts)
    rows = []
    for name in names:
        ids = tokenize(texts, name)
        lens = np.array([len(x) for x in ids])
        unk = unk_id(name)
        n_unk = sum(x.count(unk) for x in ids) if unk is not None else 0
        tok = get_tokenizer(name)
        rows.append({"tokenizer": name,
                     "vocab": tok.get_vocab_size() if _is_local(name) else len(tok),
                     "fertility": lens.sum() / n_words, "mean_tokens": lens.mean(), "max_tokens": lens.max(),
                     f"truncated@{max_len}": (lens + 2 > max_len).mean(), "unk_rate": n_unk / lens.sum(),
                     "docs_with_unk": np.mean([unk in x for x in ids]) if unk is not None else 0.0,
                     "probe_roundtrip": roundtrip(PROBE, name)})
    return pd.DataFrame(rows).set_index("tokenizer")
