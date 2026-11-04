"""Tokeniser helpers for the Name That LLM project (notebook section 2).

Three tokenisers are compared: our BPE (trained on the training split, see :func:`train_bpe`) and
two pretrained ones from the shared registry (``roberta-bpe``, ``qwen3-bpe``). All of them go
through the same preprocessing (:func:`shared.tokenization.preprocess`, NFC only) and are wrapped
in :class:`TokenizerView`, so the evaluation code treats them identically.

Typical use in a notebook::

    from tok import TokenizerView, train_bpe, describe, segmentation_table

    views = [TokenizerView.from_shared(n) for n in USED_TOKENIZERS]
    views.append(TokenizerView.from_local(train_bpe(texts, 2000), "ours-bpe-2k"))
    display(describe(views))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.tokenization import PROBE, get_tokenizer, preprocess  # noqa: E402

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
PRETRAINED = ["roberta-bpe", "qwen3-bpe"]  # = shared.tokenization.USED_TOKENIZERS
VOCAB_SIZES = [500, 1000, 2000, 4000, 8000]  # sweep for our BPE
MIN_FREQUENCY = 2
UNK_TOKEN = "[UNK]"
# Typography and punctuation the EDA found informative (checked by the cue-preservation metric)
CUE_CHARS = ["—", "–", "’", "‘", "“", "”", '"', "'", ";", ":", "-", "…"]
CUE_NAMES = {
    "—": "em-dash",
    "–": "en-dash",
    "’": "curly apostrophe",
    "‘": "curly open quote",
    "“": "curly open double",
    "”": "curly close double",
    '"': "straight double",
    "'": "straight apostrophe",
    ";": "semicolon",
    ":": "colon",
    "-": "hyphen",
    "…": "ellipsis",
}


# --------------------------------------------------------------------------- #
# Our BPE
# --------------------------------------------------------------------------- #
def make_pretokenizer(isolate_punct: bool = True):
    """Metaspace (as in the course), optionally followed by isolating every punctuation character.

    With ``isolate_punct`` each Unicode punctuation character (— ’ “ ” ; : , - ...) becomes its own
    pre-token, so BPE never merges it into a word (no ``word,`` or ``it’s`` tokens).
    """
    from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence

    if not isolate_punct:
        return Metaspace()
    return Sequence([Metaspace(), Punctuation(behavior="isolated")])


def train_bpe(
    texts,
    vocab_size: int,
    min_frequency: int = MIN_FREQUENCY,
    isolate_punct: bool = True,
):
    """Train our character-level BPE on ``texts`` (after the shared NFC preprocessing).

    Wraps the course's :func:`part1.experiment.train_tokenizer` (HF ``tokenizers`` BPE,
    ``[UNK]`` for unseen characters, alphabet = every character in ``texts``, no normaliser so case
    and typography are kept) and only swaps in :func:`make_pretokenizer`. Returns a
    ``tokenizers.Tokenizer``.
    """
    from part1.experiment import train_tokenizer

    corpus = [preprocess(t) for t in texts]
    return train_tokenizer(
        corpus,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        pretokenizer=make_pretokenizer(isolate_punct),
        unk_token=UNK_TOKEN,
    )


def bpe_name(vocab_size: int, isolate_punct: bool = True) -> str:
    """Display name for one of our BPE configurations, e.g. ``ours-bpe-2k``."""
    size = f"{vocab_size // 1000}k" if vocab_size % 1000 == 0 else str(vocab_size)
    return f"ours-bpe-{size}" + ("" if isolate_punct else "-vanilla")


CHOSEN_VOCAB_SIZES = [2000, 4000]  # decided on CV folds (notebook step 2)
COMPARISON_VOCAB_SIZE = 8000  # closest to the pretrained tokenisers' tokens per word
# Handed to the team, registered as shared.tokenization.OURS_TOKENIZERS ("ours-bpe-2k", ...)
HANDOFF_VOCAB_SIZES = CHOSEN_VOCAB_SIZES + [COMPARISON_VOCAB_SIZE]
CACHE_DIR = ROOT / "cache"  # gitignored; same folder as shared.tokenization's tutorial BPE


def final_bpe(
    vocab_size: int,
    isolate_punct: bool = True,
    min_frequency: int = MIN_FREQUENCY,
    rebuild: bool = False,
):
    """Our final BPE, trained on the whole training split; built on first use, then cached.

    Follows the shared loading protocol (:func:`shared.tokenization.get_tokenizer` does the same
    for the tutorial BPE): the JSON file in ``cache/`` is never committed, and anyone rebuilds
    it from code. Training is deterministic (same texts, in any order, give an identical
    tokeniser), so every teammate gets the same file. Returns a ``tokenizers.Tokenizer``.
    """
    from tokenizers import Tokenizer

    from shared.data import load_data

    kind = "isolated" if isolate_punct else "vanilla"
    path = CACHE_DIR / f"ours_bpe_train_vs{vocab_size}_{kind}_mf{min_frequency}.json"
    if path.exists() and not rebuild:
        return Tokenizer.from_file(str(path))
    tokenizer = train_bpe(load_data().train["text"], vocab_size, min_frequency, isolate_punct)
    CACHE_DIR.mkdir(exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


# --------------------------------------------------------------------------- #
# One interface for local and pretrained tokenisers
# --------------------------------------------------------------------------- #
@dataclass
class Encoded:
    """One tokenised text: token ids and, per token, its (start, end) span in the text."""

    text: str  # the preprocessed text the offsets refer to
    ids: list[int]
    offsets: list[tuple[int, int]]

    @property
    def pieces(self) -> list[str]:
        """Each token as the slice of text it covers (readable for byte-level tokenisers too).

        A character split over several byte-level tokens shows up in each of them.
        """
        return [self.text[s:e] for s, e in self.offsets]


@dataclass
class TokenizerView:
    """Uniform wrapper around a ``tokenizers.Tokenizer`` (local) or a Hugging Face tokeniser.

    Everything is encoded after :func:`shared.tokenization.preprocess` and without special tokens,
    matching :func:`shared.tokenization.tokenize`.
    """

    name: str
    kind: str  # "local" or "pretrained"
    tokenizer: object

    @classmethod
    def from_shared(cls, name: str) -> TokenizerView:
        """A pretrained tokeniser from the shared registry (e.g. ``roberta-bpe``)."""
        return cls(name, "pretrained", get_tokenizer(name))

    @classmethod
    def from_local(cls, tokenizer, name: str) -> TokenizerView:
        """One of our ``tokenizers.Tokenizer`` objects (e.g. from :func:`train_bpe`)."""
        return cls(name, "local", tokenizer)

    # -- properties -------------------------------------------------------- #
    @property
    def vocab_size(self) -> int:
        if self.kind == "local":
            return self.tokenizer.get_vocab_size()
        return len(self.tokenizer)  # includes added special tokens

    @property
    def unk_id(self) -> int | None:
        if self.kind == "local":
            return self.tokenizer.token_to_id(UNK_TOKEN)
        return self.tokenizer.unk_token_id

    @property
    def backend(self):
        """The underlying ``tokenizers.Tokenizer`` (pipeline: normaliser, pre-tokeniser, model)."""
        return self.tokenizer if self.kind == "local" else self.tokenizer.backend_tokenizer

    # -- encoding ---------------------------------------------------------- #
    def encode(self, texts) -> list[Encoded]:
        """Token ids and character offsets for each text."""
        texts = [preprocess(t) for t in texts]
        if self.kind == "local":
            encs = self.tokenizer.encode_batch(texts)
            return [Encoded(t, e.ids, list(e.offsets)) for t, e in zip(texts, encs)]
        out = self.tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
        return [
            Encoded(t, ids, [tuple(o) for o in offs])
            for t, ids, offs in zip(texts, out["input_ids"], out["offset_mapping"])
        ]

    def tokens(self, text: str) -> list[str]:
        """The tokeniser's own token strings (``▁``/``Ġ`` markers, byte symbols)."""
        enc = self.encode([text])[0]
        if self.kind == "local":
            return [self.tokenizer.id_to_token(i) for i in enc.ids]
        return self.tokenizer.convert_ids_to_tokens(enc.ids)

    def roundtrip(self, text: str) -> str:
        """Text as the tokeniser gives it back (encode, then decode).

        Local BPEs join their tokens and map ``▁`` to a space, as
        :func:`shared.tokenization.roundtrip` does, because ``decode`` would drop ``[UNK]``.
        """
        enc = self.encode([text])[0]
        if self.kind == "local":
            return "".join(self.tokenizer.id_to_token(i) for i in enc.ids).replace("▁", " ").strip()
        return self.tokenizer.decode(enc.ids)


# --------------------------------------------------------------------------- #
# Overviews
# --------------------------------------------------------------------------- #
def _component(obj) -> str:
    return "none" if obj is None else type(obj).__name__


def describe(views: list[TokenizerView], probe: str = PROBE) -> pd.DataFrame:
    """One row per tokeniser: vocabulary, pipeline components, and how it handles ``probe``."""
    rows = []
    for v in views:
        b = v.backend
        pre = b.pre_tokenizer
        unk = v.tokenizer.id_to_token(v.unk_id) if v.kind == "local" else v.tokenizer.unk_token
        rows.append(
            {
                "tokenizer": v.name,
                "kind": v.kind,
                "vocab_size": v.vocab_size,
                "model": _component(b.model),
                "normalizer": _component(b.normalizer),
                "pre_tokenizer": _component(pre),
                "unk_token": unk if unk is not None else "none",  # byte-level: never unknown
                "probe_tokens": len(v.encode([probe])[0].ids),
                "probe_roundtrip_ok": v.roundtrip(probe) == preprocess(probe),
            }
        )
    return pd.DataFrame(rows).set_index("tokenizer")


def segmentation_table(views: list[TokenizerView], text: str, sep: str = " | ") -> pd.DataFrame:
    """How each tokeniser splits ``text``: readable pieces and raw tokens, side by side."""
    rows = []
    for v in views:
        enc = v.encode([text])[0]
        rows.append(
            {
                "tokenizer": v.name,
                "n_tokens": len(enc.ids),
                "pieces": sep.join(enc.pieces),
                "tokens": sep.join(v.tokens(text)),
            }
        )
    return pd.DataFrame(rows).set_index("tokenizer")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
WORD_PATTERN = r"[^\W\d_]+"  # a word = a run of letters ("isn’t" -> "isn", "t")
FREQ_BUCKETS = [(0, 0, "unseen"), (1, 9, "1-9"), (10, 99, "10-99"), (100, 999, "100-999")]
FREQ_BUCKETS += [(1000, float("inf"), "1000+")]


def _overlaps(starts, ends, a: int, b: int):
    """Mask of tokens whose span overlaps the character span [a, b)."""
    return (starts < b) & (ends > a)


def _token_arrays(enc: Encoded):
    import numpy as np

    offs = np.array(enc.offsets, dtype=int).reshape(-1, 2)
    keep = offs[:, 1] > offs[:, 0]  # zero-width tokens cover no character
    return offs[keep, 0], offs[keep, 1]


def _roundtrip_texts(view: TokenizerView, encs: list[Encoded]) -> list[str]:
    if view.kind == "local":
        tok = view.tokenizer
        return ["".join(tok.id_to_token(i) for i in e.ids).replace("▁", " ").strip() for e in encs]
    return view.tokenizer.batch_decode([e.ids for e in encs])


def word_frequencies(texts) -> dict[str, int]:
    """Counts of each word (:data:`WORD_PATTERN`, case kept) in ``texts`` after preprocessing."""
    import re
    from collections import Counter

    pat = re.compile(WORD_PATTERN)
    return Counter(w for t in texts for w in pat.findall(preprocess(t)))


def _bucket(count: int) -> str:
    return next(name for lo, hi, name in FREQ_BUCKETS if lo <= count <= hi)


def evaluate_tokenizer(
    view: TokenizerView,
    texts,
    train_freq: dict[str, int] | None = None,
    cues: list[str] = CUE_CHARS,
) -> dict:
    """All four metrics of ``view`` on ``texts`` (one validation fold).

    Returns a dict with:

    - ``summary``: UNK rate (UNK tokens / tokens), share of responses with an UNK, fertility
      (tokens / whitespace word, as in :func:`shared.tokenization.tokenizer_stats`), whole-word
      rate (share of words covered by exactly one token), cue round-trip and cue distinct shares
      (over all cue occurrences).
    - ``whole_word``: whole-word rate per word-frequency bucket (frequency in ``train_freq``,
      i.e. the training fold; words never seen there are "unseen").
    - ``cues``: per cue character, occurrences, share surviving the round trip (encode, then
      decode) and share that is *distinct* (every token covering it covers only that character,
      plus whitespace; a character split into several byte-level tokens still counts).
    """
    import re

    import numpy as np

    encs = view.encode(texts)
    word_pat = re.compile(WORD_PATTERN)
    unk = view.unk_id

    n_tokens = sum(len(e.ids) for e in encs)
    n_ws_words = sum(len(e.text.split()) for e in encs)
    n_unk = sum(e.ids.count(unk) for e in encs) if unk is not None else 0
    docs_unk = np.mean([unk in e.ids for e in encs]) if unk is not None else 0.0

    bucket_words = {name: [0, 0] for *_, name in FREQ_BUCKETS}  # name -> [whole, total]
    cue_stats = {c: [0, 0, 0] for c in cues}  # char -> [occurrences, round trip, distinct]
    rts = _roundtrip_texts(view, encs)

    for e, rt in zip(encs, rts):
        starts, ends = _token_arrays(e)
        for m in word_pat.finditer(e.text):
            whole = int(_overlaps(starts, ends, m.start(), m.end()).sum() == 1)
            b = _bucket(train_freq.get(m.group(), 0)) if train_freq is not None else "1000+"
            bucket_words[b][0] += whole
            bucket_words[b][1] += 1
        for c in cues:
            pos = [i for i, ch in enumerate(e.text) if ch == c]
            if not pos:
                continue
            cue_stats[c][0] += len(pos)
            cue_stats[c][1] += min(rt.count(c), len(pos))
            for i in pos:
                cover = np.where(_overlaps(starts, ends, i, i + 1))[0]
                cue_stats[c][2] += int(
                    len(cover) > 0 and all(e.text[starts[j] : ends[j]].strip() == c for j in cover)
                )

    whole = sum(v[0] for v in bucket_words.values())
    total = sum(v[1] for v in bucket_words.values())
    occ = sum(v[0] for v in cue_stats.values())
    summary = {
        "vocab_size": view.vocab_size,
        "unk_rate": n_unk / n_tokens,
        "docs_with_unk": float(docs_unk),
        "fertility": n_tokens / n_ws_words,
        "whole_word_rate": whole / total,
        "cue_roundtrip": sum(v[1] for v in cue_stats.values()) / occ if occ else np.nan,
        "cue_distinct": sum(v[2] for v in cue_stats.values()) / occ if occ else np.nan,
    }
    ww = pd.DataFrame(
        [
            {"bucket": b, "whole_word_rate": w / n if n else np.nan, "n_words": n}
            for b, (w, n) in bucket_words.items()
        ]
    )
    cue = pd.DataFrame(
        [
            {
                "cue": c,
                "occurrences": o,
                "roundtrip": r / o if o else np.nan,
                "distinct": d / o if o else np.nan,
            }
            for c, (o, r, d) in cue_stats.items()
        ]
    )
    return {"summary": summary, "whole_word": ww, "cues": cue}


def fertility(view: TokenizerView, texts) -> float:
    """Tokens per whitespace word on ``texts`` (used for the training-fold side of the gap)."""
    encs = view.encode(texts)
    return sum(len(e.ids) for e in encs) / sum(len(e.text.split()) for e in encs)


# --------------------------------------------------------------------------- #
# Cross-validated evaluation
# --------------------------------------------------------------------------- #
def cv_evaluation(
    texts,
    folds,
    vocab_sizes: list[int] = VOCAB_SIZES,
    pretrained: list[TokenizerView] | None = None,
    min_frequency: int = MIN_FREQUENCY,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Evaluate our BPE configurations and the pretrained tokenisers on every validation fold.

    For each fold, one BPE per (vocab size, pre-tokeniser) is trained on the fold's training
    texts only; every tokeniser is then scored on the validation texts (unseen prompts) with
    :func:`evaluate_tokenizer`. Our BPEs are also scored on their own training texts
    (``fertility_train``) to measure the train-validation fertility gap.

    Returns long DataFrames ``summary`` (one row per fold x tokeniser), ``whole_word`` and
    ``cues`` (per bucket / cue character), all with ``fold``, ``tokenizer``, ``family``
    (``ours`` / ``ours-vanilla`` / pretrained name), ``vocab_target`` and ``isolate_punct``.
    """
    import time

    texts = pd.Series(list(texts))
    pretrained = pretrained or []
    out = {"summary": [], "whole_word": [], "cues": []}

    def add(res: dict, meta: dict):
        out["summary"].append({**meta, **res["summary"]})
        out["whole_word"].append(res["whole_word"].assign(**meta))
        out["cues"].append(res["cues"].assign(**meta))

    for _, k, tr, va in folds:
        t0 = time.time()
        tr_texts, va_texts = texts.iloc[tr].tolist(), texts.iloc[va].tolist()
        freq = word_frequencies(tr_texts)
        for isolate in (True, False):
            for vs in vocab_sizes:
                v = TokenizerView.from_local(
                    train_bpe(tr_texts, vs, min_frequency, isolate), bpe_name(vs, isolate)
                )
                res = evaluate_tokenizer(v, va_texts, freq)
                res["summary"]["fertility_train"] = fertility(v, tr_texts)
                meta = {
                    "fold": k,
                    "tokenizer": v.name,
                    "family": "ours" if isolate else "ours-vanilla",
                    "vocab_target": vs,
                    "isolate_punct": isolate,
                }
                add(res, meta)
        for v in pretrained:
            meta = {
                "fold": k,
                "tokenizer": v.name,
                "family": v.name,
                "vocab_target": v.vocab_size,
                "isolate_punct": False,
            }
            add(evaluate_tokenizer(v, va_texts, freq), meta)
        if verbose:
            print(f"fold {k} done in {time.time() - t0:.0f}s")

    summary = pd.DataFrame(out["summary"])
    if "fertility_train" in summary:
        summary["fertility_gap"] = summary["fertility"] - summary["fertility_train"]
    return {
        "summary": summary,
        "whole_word": pd.concat(out["whole_word"], ignore_index=True),
        "cues": pd.concat(out["cues"], ignore_index=True),
    }


def fold_summary(
    summary: pd.DataFrame, metrics: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and SD over folds per tokeniser (rows in evaluation order), as two DataFrames."""
    metrics = metrics or [
        c
        for c in [
            "vocab_size",
            "unk_rate",
            "docs_with_unk",
            "fertility",
            "fertility_train",
            "fertility_gap",
            "whole_word_rate",
            "cue_roundtrip",
            "cue_distinct",
        ]
        if c in summary
    ]
    order = summary.drop_duplicates("tokenizer").set_index("tokenizer")
    g = summary.groupby("tokenizer")[metrics]
    mean, sd = g.mean().loc[order.index], g.std().loc[order.index]
    return mean, sd


def format_mean_sd(
    mean: pd.DataFrame, sd: pd.DataFrame, decimals: dict | None = None
) -> pd.DataFrame:
    """Combine :func:`fold_summary` output into "mean ± sd" strings (SD left out when it is 0)."""
    decimals = decimals or {}
    out = pd.DataFrame(index=mean.index)
    for c in mean.columns:
        d = decimals.get(c, 3)
        out[c] = [
            f"{m:.{d}f}" if pd.isna(e) or round(e, d) == 0 else f"{m:.{d}f} ± {e:.{d}f}"
            for m, e in zip(mean[c], sd[c])
        ]
    return out


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
# Fixed colour per tokeniser family (validated categorical slots 1-3 of the dataviz reference
# palette); our vanilla ablation shares our hue and differs by line style.
COLOURS = {
    "ours": "#2a78d6",
    "ours-vanilla": "#2a78d6",
    "roberta-bpe": "#eb6834",
    "qwen3-bpe": "#1baf7a",
}
OURS_RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]  # 500 ... 8k, light -> dark
_INK, _MUTED = "#0b0b0b", "#52514e"


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def _save_show(fig, save_path, show: bool):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fertility(
    summary: pd.DataFrame,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Fertility vs. vocabulary size (log x): our BPE as curves, pretrained tokenisers as points.

    Solid = our BPE on validation folds, dotted = the same BPE on its own training folds (the gap
    shows merges fitted to the training prompts), dashed = vanilla Metaspace ablation. Error bars
    are the SD over folds.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    g = summary.groupby(["family", "vocab_target"])
    m, sd = g[["fertility", "fertility_train"]].mean(), g[["fertility", "fertility_train"]].std()

    lines = [
        ("ours", "fertility", "-", "o", "ours (validation folds)"),
        ("ours", "fertility_train", ":", "o", "ours (training folds)"),
        ("ours-vanilla", "fertility", "--", "s", "ours, vanilla Metaspace (validation)"),
    ]
    for fam, col, ls, mk, label in lines:
        x = m.loc[fam].index
        ax.errorbar(
            x,
            m.loc[fam, col],
            yerr=sd.loc[fam, col],
            color=COLOURS[fam],
            linestyle=ls,
            linewidth=2,
            marker=mk,
            markersize=6,
            capsize=3,
            label=label,
            alpha=0.55 if col == "fertility_train" else 1,
        )
    for fam, mk in (("roberta-bpe", "D"), ("qwen3-bpe", "^")):
        x, y = m.loc[fam].index[0], m.loc[(fam, m.loc[fam].index[0]), "fertility"]
        ax.errorbar(
            [x],
            [y],
            yerr=[sd.loc[(fam, x), "fertility"]],
            color=COLOURS[fam],
            marker=mk,
            markersize=9,
            linestyle="none",
            capsize=3,
            label=fam,
        )
        ax.annotate(
            f"{fam}\n{y:.2f}",
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=_INK,
        )
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary size (log scale)")
    ax.set_ylabel("fertility (tokens per whitespace word)")
    ax.set_ylim(bottom=1)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    if title:
        ax.set_title(title)
    _save_show(fig, save_path, show)
    return ax


def plot_whole_word(
    whole_word: pd.DataFrame,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Whole-word rate by word-frequency bucket (frequency in the training fold).

    Our BPE at each vocab size in a light-to-dark blue ramp (larger vocab = darker), pretrained
    tokenisers in their own colours with distinct markers. Fold means.
    """
    import matplotlib.pyplot as plt

    order = [name for *_, name in FREQ_BUCKETS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ww = whole_word.groupby(["family", "vocab_target", "bucket"])["whole_word_rate"].mean()
    ours = ww.loc["ours"]
    for colour, vs in zip(OURS_RAMP, sorted(ours.index.get_level_values(0).unique())):
        y = ours.loc[vs].reindex(order)
        ax.plot(
            order,
            y,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=6,
            label=f"ours {vs // 1000}k" if vs >= 1000 else f"ours {vs}",
        )
    for fam, mk in (("roberta-bpe", "D"), ("qwen3-bpe", "^")):
        y = ww.loc[fam].droplevel(0).reindex(order)
        ax.plot(order, y, color=COLOURS[fam], linewidth=2, marker=mk, markersize=7, label=fam)
    counts = whole_word.drop_duplicates(["fold", "bucket"]).groupby("bucket")["n_words"].mean()
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{b}\n(~{counts.get(b, 0):,.0f} words/fold)" for b in order], fontsize=8)
    ax.set_xlabel("how often the word occurs in the training fold")
    ax.set_ylabel("share of words kept as one token")
    ax.set_ylim(0, 1.03)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    if title:
        ax.set_title(title)
    _save_show(fig, save_path, show)
    return ax


def cue_table(cues: pd.DataFrame, tokenizers: list[str], value: str = "distinct") -> pd.DataFrame:
    """Tokeniser x cue character table of ``value`` (fold-pooled share), cues ordered by count.

    Shares are pooled over folds (occurrence-weighted), so rare cues are not overweighted. Columns
    are labelled "name char (n)" with the total number of occurrences.
    """
    c = cues[cues["tokenizer"].isin(tokenizers) & (cues["occurrences"] > 0)].copy()
    c["hits"] = c[value] * c["occurrences"]
    g = c.groupby(["tokenizer", "cue"])[["hits", "occurrences"]].sum()
    table = (g["hits"] / g["occurrences"]).unstack()
    n = c[c["tokenizer"] == tokenizers[0]].groupby("cue")["occurrences"].sum()
    cols = n.sort_values(ascending=False).index
    table = table.loc[tokenizers, cols]
    table.columns = [f"{CUE_NAMES.get(ch, ch)} {ch} ({n[ch]:,})" for ch in cols]
    return table


def vocab_tradeoff(
    summary: pd.DataFrame, whole_word: pd.DataFrame, family: str = "ours"
) -> pd.DataFrame:
    """One row per vocab size of our BPE: the numbers the selection rule looks at (fold means).

    - ``fertility`` and ``fertility_drop``: validation fertility and how much it falls from the
      previous (half as large) vocab size: diminishing returns show the elbow.
    - ``fertility_gap`` and ``gap_pct``: validation minus training-fold fertility, absolute and
      as % of the training fertility: merges fitted to the training prompts.
    - ``whole_word_rate`` overall and per frequency bucket for rare words.
    - ``cue_distinct``: cue preservation (must stay 1.0).
    """
    s = summary[summary["family"] == family].groupby("vocab_target")
    t = s[["fertility", "fertility_train", "fertility_gap", "whole_word_rate", "cue_distinct"]]
    t = t.mean()
    t.insert(1, "fertility_drop", -t["fertility"].diff())
    t.insert(4, "gap_pct", 100 * t["fertility_gap"] / t["fertility_train"])
    ww = whole_word[whole_word["family"] == family]
    ww = ww.groupby(["vocab_target", "bucket"])["whole_word_rate"].mean().unstack()
    for b in ["unseen", "1-9", "10-99"]:
        t[f"whole_word_{b}"] = ww[b]
    return t.drop(columns="fertility_train")
