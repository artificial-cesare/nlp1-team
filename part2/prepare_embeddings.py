"""Rebuild the step 1-2 embedding artefacts from scratch, verify them, and (optionally) re-run the probe.

Run from the repository root:

    python -m part2.prepare_embeddings                    # the chosen table only (ft-crawl, 1.5 GB download)
    python -m part2.prepare_embeddings ft-crawl glove-840b # several tables
    python -m part2.prepare_embeddings --all               # all four candidates (about 6 GB of downloads)
    python -m part2.prepare_embeddings --probe             # also re-run the step 2.5 probe, save to shared/results/

What it does, per table:
  1. downloads the official file into data/embeddings/ if missing, and checks its sha256 against the manifest;
  2. parses it once into cache/wordvecs/<name>.npy (memory-mapped afterwards);
  3. tokenises every text we will ever classify (train.csv + Kaggle test.csv; no labels used) with the table's
     agreed tokeniser (wordvecs.TOKENIZER_CONFIG) and keeps only the types they use:
     cache/wordvecs/subset-<name>.npz, about 20 MB, which is what steps 3-4 load (wordvecs.load_subset);
  4. checks that (a) the subset tokeniser gives exactly the same ids as the full-table one on every text, and
     (b) part2.clslib.EmbedAndPoolEncoder on the subset equals mean pooling on the full table;
  5. records sha256 fingerprints in shared/embeddings_manifest.json (tracked in git). If your fingerprint differs
     from the committed one, you do not have the same embeddings: the script stops and says so.

--probe re-runs the logistic-regression probe (mean-pooled vectors, raw and prompt-centred, shared 50 folds)
and saves it as shared/results/emb-<name>-mean[-promptcentred]__logreg.json, so steps 3-4 can compare
against it with shared.analysis.comparison_table. Results are deterministic (seeded folds and solver).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from part2 import wordvecs as WV  # noqa: E402

MANIFEST = ROOT / "shared" / "embeddings_manifest.json"


def sha256(path: Path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def download(name: str) -> Path:
    spec = WV.SPECS[name]
    dest = WV.EMB_DIR / spec["file"]
    if dest.exists():
        return dest
    WV.EMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {spec['file']} from {spec['url']}")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "nlp1-embeddings"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, done, t0 = int(r.headers.get("Content-Length", 0)), 0, time.time()
        while b := r.read(1 << 22):
            f.write(b)
            done += len(b)
            print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB  ({done / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s)", end="")
    print()
    tmp.rename(dest)
    return dest


def all_texts() -> list:
    from shared.data import load_data
    D = load_data()
    return D.full["text"].tolist() + pd.read_csv(WV.ROOT / "data" / "name-that-llm" / "test.csv")["text"].tolist()


def build(name: str, texts: list, manifest: dict, update: bool) -> dict:
    import torch

    from part2 import clslib

    src = download(name)
    expected = manifest.get(name, {})
    print(f"[{name}] hashing {src.name}")
    src_hash = sha256(src)
    if expected.get("source_sha256") and expected["source_sha256"] != src_hash and not update:
        raise SystemExit(f"[{name}] {src.name} differs from the manifest's file (corrupt or different release). "
                         f"Delete it and re-run, or pass --update-manifest if you mean to change it.")

    wv = WV.load(name)
    tok = WV.EmbTokenizer(wv, **WV.TOKENIZER_CONFIG[name])
    emb = tok.build_embeddings(texts)
    path = Path(str(WV.SUBSET_PATH).format(name=name))
    emb.save(path)

    # (a) the subset tokeniser reproduces the full-table ids on every text
    tok_s, emb_s = WV.load_subset(name, path)
    assert emb_s.fingerprint() == emb.fingerprint(), "save/load round trip changed the table"
    bad = sum(tok_s.ids(t) != tok.ids(t) for t in texts)
    assert bad == 0, f"{bad} texts get different ids from the subset tokeniser"
    # (b) every subset row is the full table's vector for that word
    words = emb_s.vocab.words[2:]
    rows = np.array([wv.index[w] for w in words])
    order = np.argsort(rows)
    full = np.asarray(wv.vectors[rows[order]], np.float32)[np.argsort(order)]
    assert np.array_equal(emb_s.vectors[2:], full), "subset vectors differ from the full table"
    # (c) the course encoder on the subset == mean pooling on the full table (texts without OOV tokens;
    #     word2vec has none, since punctuation is always OOV there, so (b) is its check)
    enc = clslib.EmbedAndPoolEncoder(torch.tensor(emb_s.vectors), padding_idx=emb_s.vocab.index(emb_s.pad_token))
    no_oov = [t for t in texts[:500] if all(k is not None for _, k, _ in tok.resolve(t))][:50]
    if no_oov:
        H = torch.stack([enc(torch.tensor(tok_s.ids(t))) for t in no_oov]).detach().numpy()
        assert np.allclose(H, WV.pool(tok, no_oov), atol=1e-5), "encoder output differs from full-table pooling"

    fp = emb.fingerprint()
    if expected.get("subset_fingerprint") and expected["subset_fingerprint"] != fp and not update:
        raise SystemExit(f"[{name}] subset fingerprint {fp[:12]} differs from the manifest's "
                         f"{expected['subset_fingerprint'][:12]}: different texts, tokeniser or code version.")
    print(f"[{name}] OK: {len(emb.vocab):,} types x {emb.num_dims}d -> {path.relative_to(ROOT)}  (fingerprint {fp[:12]})")
    return {"source_file": src.name, "source_url": WV.SPECS[name]["url"], "source_sha256": src_hash,
            "tokenizer_config": WV.TOKENIZER_CONFIG[name], "n_texts": len(texts), "n_types": len(emb.vocab),
            "subset_fingerprint": fp}


def probe(name: str) -> dict:
    from shared import eval_harness as H
    from shared.data import load_data
    from shared.prompt_groups import prompt_centre

    D = load_data()
    tok, _ = WV.load_subset(name)
    X = WV.pool(tok, D.train["text"].tolist())
    folds = H.load_folds(D.train["id"].tolist())
    out = {}
    for tag, Xr in [("", X), ("-promptcentred", prompt_centre(X, D.groups))]:
        res = H.run_cv(Xr, D.y, D.groups, folds, "logreg", name=f"emb-{name}-mean{tag}/logreg")
        H.save_result(res)
        out[f"probe_bal_acc{tag or '-raw'}"] = round(float(H.fold_scores(res).mean()), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="*", help=f"any of {list(WV.SPECS)} (default: {WV.CHOSEN})")
    ap.add_argument("--all", action="store_true", help="all four candidate tables")
    ap.add_argument("--probe", action="store_true", help="re-run the step 2.5 probe and save it to shared/results/")
    ap.add_argument("--update-manifest", action="store_true", help="accept new hashes (only after a deliberate change)")
    a = ap.parse_args(argv)
    names = list(WV.SPECS) if a.all else (a.tables or [WV.CHOSEN])
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    texts = all_texts()
    for name in names:
        entry = build(name, texts, manifest, a.update_manifest)
        if a.probe:
            entry |= probe(name)
            old = manifest.get(name, {})
            for k in ("probe_bal_acc-raw", "probe_bal_acc-promptcentred"):
                if k in old and abs(old[k] - entry[k]) > 1e-4 and not a.update_manifest:
                    raise SystemExit(f"[{name}] {k} = {entry[k]} but the manifest says {old[k]}: not reproduced")
        if name not in manifest or a.update_manifest:
            manifest[name] = {**manifest.get(name, {}), **entry}
        else:  # keep committed values; only fill in what was missing
            manifest[name] = {**entry, **manifest[name]}
        MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
