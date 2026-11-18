"""Static word embeddings for Name That LLM (Project 2, steps 1-2).

    from part2 import wordvecs as WV
    wv = WV.load("ft-wiki")                        # WordVectors: words, vectors (memory-mapped), lookups
    tok = WV.EmbTokenizer(wv, **WV.TOKENIZER_CONFIG["ft-wiki"])
    tok.resolve("It’s a well-known fact.")         # [(surface, key in wv or None, how), ...]
    emb = tok.build_embeddings(texts)              # small PretrainedEmbeddings-compatible table for part2.experiment

Why not gensim: it has no wheel for Python 3.14 and does not build from source here, and the two file
formats are simple enough to read with numpy. The first load parses the original file (1-3 minutes) and
caches it under cache/wordvecs/ as .npy + vocabulary; later loads memory-map the .npy in about a second.

Why not part2.clslib.PretrainedEmbeddings + BasicTokenizer (the tutorial pipeline): they lowercase the
vocabulary (keeping whichever casing comes first), lowercase the text, and NLTK rewrites quotes
(" -> `` '') and splits it’s -> it ’ s. word2vec and fastText are cased, and case, quotes and dashes are
our strongest authorship cues, so we keep the text as it is and follow each model's own token
conventions instead. `TutorialTokenizer` reproduces the tutorial pipeline as a baseline.
"""

from __future__ import annotations

import gzip
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import clslib

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"
CACHE_DIR = ROOT / "cache" / "wordvecs"

SPECS = {
    "w2v-gnews": dict(
        file="word2vec-google-news-300.bin.gz", fmt="w2v-bin",
        url="https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz",
        label="word2vec GoogleNews (skip-gram, 300d, 3M types)",
        corpus="Google News, ~100B tokens (2013)"),
    "ft-wiki": dict(
        file="wiki-news-300d-1M.vec.zip", fmt="vec",
        url="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
        label="fastText wiki-news (CBOW + subwords, 300d, 1M types)",
        corpus="Wikipedia 2017 + UMBC webbase + statmt.org news, 16B tokens"),
    "ft-crawl": dict(
        file="crawl-300d-2M.vec.zip", fmt="vec",
        url="https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
        label="fastText Common Crawl (CBOW + subwords, 300d, 2M types)",
        corpus="Common Crawl, 600B tokens"),
    "glove-840b": dict(
        file="glove.840B.300d.zip", fmt="vec",
        # nlp.stanford.edu is often down; this is the Stanford NLP group's own mirror of the same file
        url="https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip",
        label="GloVe Common Crawl 840B (cased, 300d, 2.2M types)",
        corpus="Common Crawl, 840B tokens"),
}

# Tokeniser settings per table, read off each vocabulary in step 1.2 (cesare_word_embeddings.ipynb) and checked
# against coverage (1.4) and a linear probe (2.5). Use these in steps 3-4 so everyone tokenises the same way.
TOKENIZER_CONFIG = {
    "w2v-gnews": dict(clitics="backoff", hyphens="backoff", digits="hash", phrases=3, curly="fold", case_backoff="lower"),
    "ft-wiki": dict(clitics="split", hyphens="backoff", digits="keep", phrases=1, curly="mark", case_backoff="lower"),
    "ft-crawl": dict(clitics="split", hyphens="backoff", digits="keep", phrases=1, curly="mark", case_backoff="lower"),
    # GloVe 840B has the curly clitics (’s, n’t) as their own types, so curly='keep' already keeps the cue
    "glove-840b": dict(clitics="ptb", hyphens="backoff", digits="keep", phrases=1, curly="keep", case_backoff="lower"),
}
CHOSEN = "ft-crawl"  # best in the step 2.5 probe; glove-840b is a close second (-1.6 pp raw, -1.9 pp prompt-centred)

# Typographic cues from the part 1 EDA, with the "plain" variant each one competes with.
CUES = {
    "—": "em dash", "–": "en dash", "-": "hyphen", "--": "double hyphen",
    "’": "curly apostrophe", "'": "straight apostrophe",
    "“": "curly open quote", "”": "curly close quote", '"': "straight quote",
    ";": "semicolon", ":": "colon", ",": "comma", ".": "full stop",
    "(": "parenthesis", "…": "ellipsis", "...": "three dots", "?": "question mark", "!": "exclamation",
}

# Lookup-only normalisation: the surface token keeps its form, only the key used for the table changes.
QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "−": "-"})


# ----------------------------------------------------------------------------------------------- loading
def _read_vec(path: Path, verbose=True):
    """Text vectors, one 'word v1 ... vD' line per type, plain or zipped: fastText .vec (header line 'V D')
    or GloVe .txt (no header)."""
    def lines():
        if path.suffix == ".zip":
            with ZipFile(path) as z, z.open(z.namelist()[0]) as f:
                yield from TextIOWrapper(f, encoding="utf-8", newline="\n")
        else:
            with open(path, encoding="utf-8", newline="\n") as f:
                yield from f

    it = lines()
    first = next(it)
    head = first.split()
    if len(head) == 2:                       # fastText header
        n, dim = map(int, head)
        pending = []
    else:                                    # GloVe: no header, the first line is already a vector
        n, dim = None, len(first.rstrip("\n").rstrip(" ").split(" ")) - 1
        pending = [first]
    words, chunks, batch = [], [], []
    row = 0

    def flush():
        nonlocal row, batch
        if batch:
            chunks.append(np.fromstring(" ".join(batch), sep=" ", dtype=np.float32).reshape(len(batch), dim))
            row += len(batch)
            batch = []

    for line in _chain(pending, it):
        parts = line.rstrip("\n").rstrip(" ").rsplit(" ", dim)
        if len(parts) != dim + 1 or "\n" in parts[0]:
            continue  # malformed line (a handful in these files)
        words.append(parts[0])
        batch.append(" ".join(parts[1:]))
        if len(batch) == 50_000:
            flush()
            if verbose:
                print(f"\r  {row:,}/{n or 0:,}", end="")
    flush()
    if verbose:
        print()
    return words, np.concatenate(chunks)


def _chain(*its):
    for it in its:
        yield from it


def _read_w2v_bin(path: Path, verbose=True):
    """Original word2vec binary format: header 'V D\\n', then per type: word bytes, ' ', D float32."""
    with gzip.open(path, "rb") as f:
        buf = f.read()
    end = buf.index(b"\n")
    n, dim = map(int, buf[:end].split())
    vecs, words = np.empty((n, dim), np.float32), []
    pos, nbytes = end + 1, 4 * dim
    for i in range(n):
        sp = buf.index(b" ", pos)
        words.append(buf[pos:sp].lstrip(b"\n").decode("utf-8", errors="replace"))
        vecs[i] = np.frombuffer(buf, np.float32, dim, sp + 1)
        pos = sp + 1 + nbytes
        if verbose and i % 500_000 == 0:
            print(f"\r  {i:,}/{n:,}", end="")
    if verbose:
        print()
    return words, vecs


@dataclass
class WordVectors:
    """A pretrained table: `words[i]` has vector `vectors[i]`. Rows are in file order, which for both
    word2vec and fastText is descending corpus frequency, so a word's row index is its frequency rank."""
    name: str
    words: list
    vectors: np.ndarray
    top_n: int = 200_000          # nearest-neighbour searches use the most frequent types only (standard practice)
    index: dict = field(init=False, repr=False)
    _unit_top: np.ndarray = field(init=False, repr=False, default=None)
    _lower: dict = field(init=False, repr=False, default=None)

    def __post_init__(self):
        self.index = {}
        for i, w in enumerate(self.words):
            self.index.setdefault(w, i)   # first (most frequent) occurrence wins

    def __contains__(self, w):
        return w in self.index

    def __len__(self):
        return len(self.words)

    def __repr__(self):
        return f"WordVectors({self.name!r}, {len(self):,} types x {self.dim}d)"

    @property
    def dim(self):
        return self.vectors.shape[1]

    @property
    def lower_index(self) -> dict:
        """Lowercased key -> row of its most frequent casing (what the tutorial's PretrainedEmbeddings keeps)."""
        if self._lower is None:
            self._lower = {}
            for i, w in enumerate(self.words):
                self._lower.setdefault(w.lower(), i)
        return self._lower

    def rank(self, w):
        return self.index.get(w)

    def vec(self, w) -> np.ndarray:
        return np.asarray(self.vectors[self.index[w]], dtype=np.float32)

    @property
    def unit_top(self) -> np.ndarray:
        if self._unit_top is None:
            V = np.asarray(self.vectors[: self.top_n], dtype=np.float32)
            self._unit_top = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
        return self._unit_top

    def similarity(self, a: str, b: str) -> float:
        return cos(self.vec(a), self.vec(b))

    def most_similar(self, query, k=10, exclude=()):
        """Top-k types among the `top_n` most frequent by cosine. `query` is a word or a vector."""
        if isinstance(query, str):
            exclude = set(exclude) | {query}
            query = self.vec(query)
        q = np.asarray(query, np.float32)
        s = self.unit_top @ (q / max(np.linalg.norm(q), 1e-8))
        out = []
        for i in np.argsort(-s):
            if self.words[i] not in exclude:
                out.append((self.words[i], float(s[i])))
            if len(out) == k:
                break
        return out

    def analogy(self, a, b, c, k=5):
        """a - b + c, e.g. analogy('king', 'man', 'woman') -> queen."""
        return self.most_similar(self.vec(a) - self.vec(b) + self.vec(c), k, exclude={a, b, c})


def cos(u, v) -> float:
    return float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))


def load(name: str, verbose=True) -> WordVectors:
    """Load one of SPECS (download it first, see SPECS[name]['url']); cached as memory-mapped .npy."""
    spec = SPECS[name]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy, txt = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}.vocab.txt"
    t0 = time.time()
    if not npy.exists():
        src = EMB_DIR / spec["file"]
        if not src.exists():
            raise FileNotFoundError(f"{src} missing; download it with:\n  curl -L -o {src} {spec['url']}")
        if verbose:
            print(f"parsing {src.name} (first time only)")
        words, vecs = (_read_w2v_bin if spec["fmt"] == "w2v-bin" else _read_vec)(src, verbose)
        np.save(npy, vecs)
        txt.write_text("\n".join(words), encoding="utf-8")
        del vecs
    words = txt.read_text(encoding="utf-8").split("\n")
    wv = WordVectors(name, words, np.load(npy, mmap_mode="r"))
    if verbose:
        print(f"{wv}  loaded in {time.time() - t0:.0f}s")
    return wv


# --------------------------------------------------------------------------------------------- tokenising
# Words (letters/digits, with inner apostrophes, hyphens or dots as in "don't", "well-known", "U.S.", "3.5"),
# then any single non-space character on its own (every punctuation mark is its own token).
_WORD = r"\w+(?:[-'’.]\w+)*"
_TOKEN_RE = re.compile(rf"\.\.\.|--|{_WORD}|[^\w\s]")
# fastText's corpora were split at the apostrophe: don't -> don 't, it's -> it 's, I'm -> I 'm
_CLITIC_RE = re.compile(r"^(.+?)(['’](?:t|s|re|ve|ll|d|m))$", re.IGNORECASE)
# Penn Treebank (Stanford tokeniser, used for GloVe): don't -> do n't, it's -> it 's
_PTB_RE = re.compile(r"^(.+?)(n['’]t|['’](?:s|re|ve|ll|d|m))$", re.IGNORECASE)
_CURLY = set("’‘“”")


class EmbTokenizer(clslib.Tokenizer):
    """Case-preserving tokeniser whose token definition is adapted to one embedding table.

    Options (each mirrors a convention of the table's training corpus; step 1 reads them off the vocabulary
    and checks them against coverage):
      clitics      'whole' keeps "don't" as one token; 'split' gives "don 't" / "it 's" (fastText convention);
                   'ptb' gives "do n't" / "it 's" (Penn Treebank, the Stanford tokeniser behind GloVe);
                   'backoff' keeps the whole word if the table has it, else splits (word2vec has don't but not it's).
      hyphens      'keep' keeps "well-known"; 'backoff' keeps it if the table has it, else tries well_known,
                   else splits at the hyphens.
      digits       'keep', or 'hash' to replace every digit with '#' (word2vec GoogleNews convention).
      phrases      longest n-gram (joined by '_') merged when the table has it, e.g. New_York (word2vec); 1 = off.
      curly        how to treat ’ ‘ “ ” – − in tokens the table lacks (e.g. it’s -> 's is missing, 's is not):
                   'keep'  no change, the token stays OOV;
                   'fold'  look up the ASCII form instead ('s); the curly/straight distinction is lost;
                   'mark'  as 'fold', and also emit the curly character as its own token (’ has a vector),
                           so the cue survives pooling.
      case_backoff if still missing: 'lower' retries lowercased (sentence-initial words); 'both' also retries
                   capitalised (this is how word2vec's removed a/and/of/to would come back, as A/And/Of/To);
                   'none' disables it.

    `resolve(text)` returns (surface token, key in the table or None, how) with how in
    {'exact', 'fold', 'case', 'oov', 'mark'}; `tokens` / `ids` give the part2.clslib.Tokenizer interface.
    """

    def __init__(self, wv: WordVectors, clitics="split", hyphens="backoff", digits="keep", phrases=1,
                 curly="fold", case_backoff="lower"):
        assert clitics in ("whole", "split", "backoff", "ptb") and curly in ("keep", "fold", "mark")
        assert case_backoff in ("none", "lower", "both")
        self.wv, self.clitics, self.hyphens, self.digits = wv, clitics, hyphens, digits
        self.phrases, self.curly, self.case_backoff = phrases, curly, case_backoff
        self.vocab = None  # a clslib.Vocabulary, set by build_embeddings()
        self._found = None  # while building a subset: every key a lookup found, used in the end or not

    @property
    def config(self) -> dict:
        return dict(clitics=self.clitics, hyphens=self.hyphens, digits=self.digits, phrases=self.phrases,
                    curly=self.curly, case_backoff=self.case_backoff)

    def _norm(self, t):
        return re.sub(r"\d", "#", t) if self.digits == "hash" else t

    def _surface(self, text):
        out = []
        for t in _TOKEN_RE.findall(text):
            m = (_PTB_RE if self.clitics == "ptb" else _CLITIC_RE).match(t) if self.clitics != "whole" else None
            if m and (self.clitics in ("split", "ptb") or self._key(t)[0] is None):
                out += [m.group(1), m.group(2)]
            elif "-" in t.strip("-") and self.hyphens == "backoff" and self._key(t)[0] is None:
                if self._key(t.replace("-", "_"))[0] is not None:
                    out.append(t.replace("-", "_"))
                    continue
                parts = t.split("-")
                for j, p in enumerate(parts):
                    out += ([p] if p else []) + (["-"] if j < len(parts) - 1 else [])
            else:
                out.append(t)
        return out

    def _key(self, t):
        """(key, how) for one surface token."""
        key, how = self._lookup(t)
        if key is not None and self._found is not None:
            self._found.add(key)
        return key, how

    def _lookup(self, t):
        t = self._norm(t)
        if t in self.wv:
            return t, "exact"
        f = t.translate(QUOTE_FOLD) if self.curly != "keep" else t
        if f != t and f in self.wv:
            return f, "fold"
        if self.case_backoff != "none":
            for c in (f.lower(), f.capitalize())[: 1 if self.case_backoff == "lower" else 2]:
                if c != f and c in self.wv:
                    return c, "case"
        return None, "oov"

    def resolve(self, text: str):
        toks = self._surface(text)
        out, i = [], 0
        while i < len(toks):
            for n in range(min(self.phrases, len(toks) - i), 1, -1):  # greedy longest phrase first
                cand = "_".join(self._norm(t) for t in toks[i:i + n])
                if cand in self.wv:
                    if self._found is not None:
                        self._found.add(cand)
                    out.append((" ".join(toks[i:i + n]), cand, "exact"))
                    i += n
                    break
            else:
                key, how = self._key(toks[i])
                if self.curly == "mark" and how in ("fold", "case") and toks[i] != key:
                    out += [(c, c, "mark") for c in toks[i] if c in _CURLY and c in self.wv]
                out.append((toks[i], key, how))
                i += 1
        return out

    def tokens(self, text: str):
        return [s for s, _, _ in self.resolve(text)]

    def keys(self, text: str, unk="<unk>"):
        return [k if k is not None else unk for _, k, _ in self.resolve(text)]

    def ids(self, text: str):
        if self.vocab is None:
            raise ValueError("call build_embeddings(texts) first: it fixes the (small) vocabulary")
        return [self.vocab.index(k) for k in self.keys(text)]

    def build_embeddings(self, texts) -> "SubsetEmbeddings":
        """A table restricted to the types these texts use (plus <pad>, <unk>), for part2.experiment.

        Restricting is only a speed/memory measure (a 3M x 300 nn.Embedding is 3.6 GB); vectors are unchanged.
        Pass every text the model will ever see (train + test; no labels are used), else unseen types map to <unk>.
        The subset keeps every key a lookup *found* while tokenising, not only the final ones: tokenising decisions
        depend on intermediate lookups (e.g. double-hulled -> double_hulled, later merged into a phrase), and with
        all of them present a tokeniser over the subset makes exactly the same decisions as over the full table.
        """
        self._found = set()
        try:
            for t in texts:
                self.resolve(t)
            keys = sorted(self._found, key=self.wv.rank)
        finally:
            self._found = None
        emb = SubsetEmbeddings.from_table(self.wv, keys)
        self.vocab = emb.vocab
        return emb


class TutorialTokenizer(clslib.Tokenizer):
    """The tutorial's pipeline as a baseline: lowercase + NLTK word_tokenize, looked up in a lowercased table
    (PretrainedEmbeddings lowercases every key and keeps the first, i.e. most frequent, casing)."""

    def __init__(self, wv: WordVectors):
        self.wv = wv
        self._bt = clslib.BasicTokenizer(lowercase=True, remove_punct=False, remove_stopwords=False)

    def resolve(self, text):
        out = []
        for t in self._bt.tokens(text):
            i = self.wv.lower_index.get(t)
            out.append((t, self.wv.words[i] if i is not None else None, "exact" if i is not None else "oov"))
        return out

    def tokens(self, text):
        return self._bt.tokens(text)


class SubsetEmbeddings(clslib.PretrainedEmbeddings):
    """Drop-in for part2.clslib.PretrainedEmbeddings (same attributes: vocab, vectors, pad_token, unk_token),
    but case-sensitive and holding only the rows we need. <unk> is the mean of the full table, as in clslib."""

    def __init__(self, words, vectors, unk_vector, pad_token="<pad>", unk_token="<unk>"):
        self.pad_token, self.unk_token, self.num_dims = pad_token, unk_token, vectors.shape[1]
        self.vocab = clslib.Vocabulary([pad_token, unk_token] + list(words), default=unk_token, frozen=True)
        self.vectors = np.vstack([np.zeros((1, self.num_dims), np.float32), unk_vector[None].astype(np.float32),
                                  np.asarray(vectors, np.float32)])

    @classmethod
    def from_table(cls, wv: WordVectors, keys):
        rows = np.array([wv.index[k] for k in keys], dtype=np.int64)
        # exact mean of the whole table, in chunks so the memory-mapped matrix is never fully in RAM
        unk = sum(np.asarray(wv.vectors[i:i + 500_000], np.float64).sum(0) for i in range(0, len(wv), 500_000)) / len(wv)
        order = np.argsort(rows)
        V = np.asarray(wv.vectors[rows[order]], np.float32)[np.argsort(order)]  # memmap indexing needs sorted rows
        return cls(keys, V, unk)

    def pca(self, k: int):
        """Same vocabulary, vectors projected on their top-k principal components (fitted on these types only,
        no labels). The official fastText/GloVe releases are 300d only, so this is the embedding-size axis of
        step 3 with corpus, algorithm and tokenisation held fixed."""
        V = np.asarray(self.vectors[2:], np.float64)
        mu = V.mean(0)
        _, _, Vt = np.linalg.svd(V - mu, full_matrices=False)
        P = Vt[:k].T
        return SubsetEmbeddings(self.vocab.words[2:], ((V - mu) @ P).astype(np.float32),
                                ((self.vectors[1] - mu) @ P).astype(np.float32))

    def random_like(self, seed=42):
        """Same vocabulary, random N(0, sd) vectors with the pretrained table's per-dimension sd (pad stays 0).
        For step 4's 'trained or not' axis: pretrained vs random init, each frozen or trainable."""
        rng = np.random.default_rng(seed)
        sd = self.vectors[2:].std(0)
        V = (rng.standard_normal(self.vectors[2:].shape) * sd).astype(np.float32)
        return SubsetEmbeddings(self.vocab.words[2:], V, (rng.standard_normal(self.num_dims) * sd).astype(np.float32))

    def __contains__(self, word):
        return word in self.vocab

    def index(self, word):
        return self.vocab.index(word)

    def embed_word(self, word):
        return self.vectors[self.index(word)]

    def fingerprint(self) -> str:
        """sha256 of the vocabulary and vectors (not of the .npz, whose zip timestamps change on every save).
        Equal fingerprints = identical tables; shared/embeddings_manifest.json records the expected ones."""
        import hashlib
        h = hashlib.sha256("\n".join(self.vocab.words).encode("utf-8"))
        h.update(np.ascontiguousarray(self.vectors, dtype=np.float32).tobytes())
        return h.hexdigest()

    def save(self, path):
        np.savez_compressed(path, words=np.array(self.vocab.words[2:], dtype=str), vectors=self.vectors[2:],
                            unk=self.vectors[1])

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls([str(w) for w in z["words"]], z["vectors"], z["unk"])


# ---------------------------------------------------------------------------------------------- coverage
def coverage(tokenizer, texts) -> dict:
    """Token-level and type-level coverage of `texts` under `tokenizer.resolve`."""
    how, types, oov_types, n_tok, texts_with_oov = Counter(), set(), Counter(), [], 0
    word_tok = word_oov = 0
    for t in texts:
        r = [x for x in tokenizer.resolve(t) if x[2] != "mark"]  # inserted cue tokens are not text tokens
        n_tok.append(len(r))
        has_oov = False
        for s, k, h in r:
            how[h] += 1
            types.add(s)
            is_word = bool(re.search(r"\w", s))
            word_tok += is_word
            if k is None:
                oov_types[s] += 1
                has_oov = True
                word_oov += is_word
        texts_with_oov += has_oov
    total = sum(how.values())
    return {
        "token coverage": 1 - how["oov"] / total,
        "word-token coverage": 1 - word_oov / word_tok,
        "exact": how["exact"] / total, "via curly fold": how["fold"] / total, "via case backoff": how["case"] / total,
        "type coverage": 1 - len(oov_types) / len(types),
        "texts with OOV": texts_with_oov / len(texts),
        "tokens / text": float(np.mean(n_tok)),
        "_oov": oov_types,
    }


def coverage_table(rows: dict) -> pd.DataFrame:
    """rows: {name: coverage(...)} -> DataFrame without the private OOV counters."""
    return pd.DataFrame({k: {m: v for m, v in c.items() if not m.startswith("_")} for k, c in rows.items()}).T


def coverage_by_class(tokenizer, texts, labels) -> pd.Series:
    """OOV rate (share of tokens not in the table) per class label."""
    df = pd.DataFrame({"y": list(labels), "r": [[x for x in tokenizer.resolve(t) if x[2] != "mark"] for t in texts]})
    oov = df["r"].map(lambda r: sum(k is None for _, k, _ in r))
    n = df["r"].map(len)
    return (oov.groupby(df["y"]).sum() / n.groupby(df["y"]).sum()).rename("OOV rate")


# -------------------------------------------------------------------------------------------- representations
def pool(tokenizer, texts, how="mean", keep=None) -> np.ndarray:
    """Embed-and-pool with the full table: mean (or sum) of the in-vocabulary token vectors (OOV skipped).
    keep: optional predicate on the surface token, to pool only some tokens (e.g. only function words)."""
    wv = tokenizer.wv
    out = np.zeros((len(texts), wv.dim), np.float32)
    for n, t in enumerate(texts):
        rows = sorted(wv.index[k] for s, k, _ in tokenizer.resolve(t) if k is not None and (keep is None or keep(s)))
        if rows:
            V = np.asarray(wv.vectors[rows], np.float32)
            out[n] = V.mean(0) if how == "mean" else V.sum(0)
    return out


def variance_explained(X, groups) -> float:
    """Between-group share of the total variance (sum over dimensions): eta^2 of a one-way ANOVA."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    g = pd.Series(np.asarray(groups))
    means = pd.DataFrame(X).groupby(g.values).transform("mean").to_numpy()
    return float(((means - mu) ** 2).sum() / ((X - mu) ** 2).sum())


def log_odds(counts_a: Counter, counts_b: Counter, prior: Counter, alpha0=None) -> pd.Series:
    """Weighted log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008), as z-scores.
    Positive = more typical of a than of b."""
    alpha0 = alpha0 or sum(prior.values())
    # totals over the prior's vocabulary only: mixing vocabularies biases every word towards the smaller group
    n_a, n_b = sum(counts_a.get(w, 0) for w in prior), sum(counts_b.get(w, 0) for w in prior)
    n_p = sum(prior.values())
    z = {}
    for w, p in prior.items():
        a_w = alpha0 * p / n_p
        ya, yb = counts_a.get(w, 0), counts_b.get(w, 0)
        d = np.log((ya + a_w) / (n_a + alpha0 - ya - a_w)) - np.log((yb + a_w) / (n_b + alpha0 - yb - a_w))
        z[w] = d / np.sqrt(1 / (ya + a_w) + 1 / (yb + a_w))
    return pd.Series(z).sort_values(ascending=False)


def mean_pairwise_cos(wv: WordVectors, words) -> float:
    V = np.stack([wv.vec(w) for w in words])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    n = len(words)
    return float((S.sum() - n) / (n * (n - 1)))


# ------------------------------------------------------------------------------------------------ hand-off
SUBSET_PATH = CACHE_DIR / "subset-{name}.npz"


def load_subset(name: str, path=None):
    """(tokenizer, embeddings) for steps 3-4 from the small subset file written by part2/prepare_embeddings.py.

    No full table needed: every lookup key our texts can resolve to is in the subset, so this tokeniser gives
    exactly the same ids as one built on the full table (prepare_embeddings checks it on every text).
        tok, emb = WV.load_subset("ft-crawl")
        part2.experiment.train(train_pairs, val_pairs, tok, emb, num_classes=18, enc_type="embed-and-pool", ...)
    """
    path = Path(path or str(SUBSET_PATH).format(name=name))
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run  python -m part2.prepare_embeddings {name}")
    emb = SubsetEmbeddings.load(path)
    wv = WordVectors(name, emb.vocab.words[2:], emb.vectors[2:])
    tok = EmbTokenizer(wv, **TOKENIZER_CONFIG[name])
    tok.vocab = emb.vocab
    return tok, emb
