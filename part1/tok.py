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
