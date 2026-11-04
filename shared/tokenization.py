"""Shared preprocessing and tokenisation for NLP1 Project 1 (Name That LLM).

    from shared.tokenization import preprocess, tokenize, tokenizer_stats, TOKENIZERS
    ids = tokenize(texts, "roberta-bpe")          # list of token-id lists
    tokenizer_stats(texts)                        # fertility, truncation@512, unk rate, typography round trip

Preprocessing is NFC only (no lowercasing, no stopword or punctuation removal):
punctuation and typography are authorship cues. See project.ipynb, steps 3-4.
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

TOKENIZERS = {  # name -> Hugging Face id (or "tutorial")
    "roberta-bpe": "rrivera1849/LUSR",          # LUSR; provisional vocabulary for the from-scratch transformer
    "qwen3-bpe": "Qwen/Qwen3-Embedding-0.6B",   # identical to Qwen/Qwen3-0.6B-Base's tokeniser
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta-v3": "microsoft/deberta-v3-base",
    "tutorial-bpe-1k": "tutorial",              # part1.experiment.train_tokenizer on AG News, vocab 1000, min_freq 2
}

PROBE = "He said “no” — it’s 5 µm…  done"  # curly quotes, em dash, µ, ellipsis, double space


def preprocess(text: str) -> str:
    """NFC only. NFKC would rewrite superscripts, '…' and non-breaking spaces, erasing cues."""
    return unicodedata.normalize("NFC", text)


def _is_tutorial(name: str) -> bool:
    return TOKENIZERS[name] == "tutorial"


@lru_cache(maxsize=None)
def get_tokenizer(name: str):
    if not _is_tutorial(name):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZERS[name])
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
    if _is_tutorial(tokenizer_name):
        return [e.ids for e in tok.encode_batch(texts)]
    return tok(texts, add_special_tokens=False)["input_ids"]


def unk_id(tokenizer_name: str):
    tok = get_tokenizer(tokenizer_name)
    return tok.token_to_id("[UNK]") if _is_tutorial(tokenizer_name) else tok.unk_token_id


def roundtrip(text: str, tokenizer_name: str) -> str:
    """Text as the tokeniser sees it. For the tutorial BPE we join tokens, since its decode drops [UNK]."""
    tok = get_tokenizer(tokenizer_name)
    if _is_tutorial(tokenizer_name):
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
                     "vocab": tok.get_vocab_size() if _is_tutorial(name) else len(tok),
                     "fertility": lens.sum() / n_words, "mean_tokens": lens.mean(), "max_tokens": lens.max(),
                     f"truncated@{max_len}": (lens + 2 > max_len).mean(), "unk_rate": n_unk / lens.sum(),
                     "docs_with_unk": np.mean([unk in x for x in ids]) if unk is not None else 0.0,
                     "probe_roundtrip": roundtrip(PROBE, name)})
    return pd.DataFrame(rows).set_index("tokenizer")
