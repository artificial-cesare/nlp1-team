"""Cross-validate the course's neural classifiers (part2.experiment) on the team's shared protocol.

Use this for steps 3-6 instead of part2.experiment.cross_validation, which
  - picks the best epoch on the same fold it then reports (selection on the test fold: optimistic),
  - uses folds that are not grouped by prompt (answers to one prompt leak between training and test),
  - reports plain accuracy only.

Here, for every shared fold (shared/folds.json, grouped by prompt):
  1. seed everything with SEED + repeat (results are reproducible on CPU);
  2. split the outer training fold by prompt into inner train / inner validation (shared.eval_harness.inner_split);
  3. train with part2.experiment.train on inner train, keeping the epoch with the best inner-validation loss;
  4. predict the untouched outer test fold and score it (balanced accuracy first, as everywhere else).
The result has the same format as shared.eval_harness.run_cv, so H.save_result, H.compare and
shared.analysis.results_table / comparison_table work unchanged, e.g. against the step 2.5 probe
"emb-ft-crawl-mean/logreg" that prepare_embeddings saves.

    from part2 import embcv, wordvecs as WV
    from shared import eval_harness as H
    tok, emb = WV.load_subset("ft-crawl")
    res = embcv.run_cv_torch(tok, emb, "embed-and-pool", name="ft-crawl-pool/mlp-50x50",
                             regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
                             opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20})
    H.save_result(res)

Difference from the harness's own MLPs: those refit on the whole outer training fold after choosing the epoch;
here the final model is the inner-train one (20% less data, but no second training run). State it if you compare.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
import time
from pathlib import Path

import numpy as np

from shared import eval_harness as H
from shared.data import load_data

from . import experiment


def run_cv_torch(tok, emb, enc_type: str, name: str, enc_kwargs=None, regressor_kwargs=None, opt_kwargs=None,
                 repeats: int = 1, folds=None, device: str = "cpu", workdir=None, verbose: bool = False) -> dict:
    """Nested CV of one part2 neural classifier. `tok`, `emb` from wordvecs.load_subset (or any course pair).

    repeats: how many of the 10 shared CV repeats to use (1 = 5 folds; neural models are slow).
    folds: override with your own list of (repeat, fold, train_idx, test_idx), e.g. FOLDS[:1] for a smoke test.
    """
    D = load_data()
    texts, y, groups = D.train["text"].tolist(), D.y, D.groups
    folds = folds if folds is not None else H.load_folds(D.train["id"].tolist(), repeats=repeats)
    enc_kwargs = enc_kwargs or {"freeze_embeddings": True}
    regressor_kwargs = regressor_kwargs or {"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1}
    opt_kwargs = opt_kwargs or {"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20}
    workdir = Path(workdir or tempfile.mkdtemp(prefix="embcv-"))
    workdir.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(texts, y))
    n_classes = len(D.classes)

    rows, oof_pred, oof_proba = [], np.full(len(y), -1), np.zeros((len(y), n_classes))
    t0 = time.time()
    for r, k, tr, te in folds:
        H.set_seed(H.SEED + r)
        itr, iva = H.inner_split(np.asarray(tr), groups, seed=H.SEED + r)
        path = workdir / f"r{r}-k{k}"
        quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with quiet:
            metrics, best = experiment.train(
                [pairs[i] for i in itr], [pairs[i] for i in iva], tok, emb, num_classes=n_classes,
                enc_type=enc_type, save_to=str(path),
                enc_kwargs=dict(enc_kwargs), regressor_kwargs=dict(regressor_kwargs), opt_kwargs=dict(opt_kwargs),
                device=device)
        model = experiment.load(str(path), emb, device)
        logp = experiment.predict_logprobs(model, tok, [texts[i] for i in te], 256, device)
        pred = logp.argmax(1)
        rows.append({"repeat": r, "fold": k, **H.scores(y[te], pred), "best_epoch": int(best),
                     "inner_val_loss": float(metrics[best]["val_metrics"]["loss"]),
                     "n_test": len(te), "n_total": len(tr) + len(te)})
        if r == folds[0][0]:
            oof_pred[te], oof_proba[te] = pred, np.exp(logp)
        if verbose:
            print(f"repeat {r} fold {k}: bal.acc {rows[-1]['balanced_accuracy']:.4f} (best epoch {best})")
    res = {"name": name, "model": f"torch-{enc_type}", "folds": rows, "oof_pred": oof_pred, "oof_proba": oof_proba,
           "seconds": time.time() - t0, "runs": len({f[0] for f in folds}),
           "config": {"enc_type": enc_type, "enc_kwargs": enc_kwargs, "regressor_kwargs": regressor_kwargs,
                      "opt_kwargs": opt_kwargs}}
    ba = H.fold_scores(res)
    print(f"{name:28s} bal.acc {ba.mean():.4f} ± {ba.std():.4f}  ({len(rows)} folds, {res['seconds']:.0f}s)")
    return res
