"""Shared evaluation harness for NLP1 Project 1 (Name That LLM).

Every representation (token indicators, bag-of-words, neural embeddings) goes
through this file, so all comparisons use the same folds, models, grids and
Bayesian test. Input: a feature matrix X (dense numpy or scipy sparse) whose rows
are aligned with Kazik's train split (`data/name-that-llm/split_train.csv`).

Protocol (see part1/neural_project.ipynb, step 2):
- Folds: StratifiedGroupKFold(5) grouped by prompt_id, repeated R times
  (different shuffles). Saved in shared/folds.json, keyed by passage id.
- Tuning: nested. Inside each outer training fold we hold out 20% of the
  prompts (grouped) as an inner validation set, pick the grid point with the best
  inner balanced accuracy, then refit on the full outer training fold. The
  neural models use that same inner split for early stopping and are refit for
  the chosen number of epochs.
- Metric: balanced accuracy (Kaggle metric); accuracy and macro-F1 also reported.
- Comparison: Bayesian correlated t-test (Benavoli et al. 2017), rho = n_test / n_total.

Usage (any representation X whose rows align with shared.data.load_data().train):
    from shared.data import load_data
    from shared import eval_harness as H
    D = load_data(); FOLDS = H.load_folds(D.train["id"].tolist())
    res = H.run_cv(X, D.y, D.groups, FOLDS, "logreg", name="bow/logreg"); H.save_result(res)
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.stats as st
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
ROPE = 0.01  # 1 percentage point of balanced accuracy (justified in the notebook)
FOLDS_PATH = Path(__file__).with_name("folds.json")


def set_seed(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- folds
def make_folds(ids, y, groups, repeats: int, n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """Repeated StratifiedGroupKFold. Returns {"repeats": [[{"train": ids, "test": ids}, ...], ...]}."""
    ids, y, groups = np.asarray(ids), np.asarray(y), np.asarray(groups)
    out = []
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        out.append([{"train": ids[tr].tolist(), "test": ids[te].tolist()} for tr, te in cv.split(ids, y, groups)])
    return {"seed": seed, "n_splits": n_splits, "grouped_by": "prompt_id", "repeats": out}


def save_folds(folds: dict, path: Path = FOLDS_PATH) -> None:
    path.write_text(json.dumps(folds))


def load_folds(ids, repeats: int | None = None, path: Path = FOLDS_PATH):
    """Load folds as positional indices into `ids` (the train split's id column).

    Returns a list of (repeat, fold, train_idx, test_idx). `repeats` truncates to the first R repeats.
    """
    folds = json.loads(Path(path).read_text())
    pos = {pid: i for i, pid in enumerate(ids)}
    assert len(pos) == len(ids), "ids must be unique"
    reps = folds["repeats"][: repeats or len(folds["repeats"])]
    return [
        (r, k, np.array([pos[i] for i in f["train"]]), np.array([pos[i] for i in f["test"]]))
        for r, rep in enumerate(reps)
        for k, f in enumerate(rep)
    ]


def inner_split(train_idx, groups, seed: int = SEED, val_frac: float = 0.2):
    """Grouped holdout inside an outer training fold (for tuning + early stopping)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(train_idx, groups=np.asarray(groups)[train_idx]))
    return train_idx[tr], train_idx[va]


# --------------------------------------------------------------------------- metrics
def scores(y_true, y_pred) -> dict:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# --------------------------------------------------------------------------- torch models
class MLP(nn.Module):
    """Single hidden layer: Linear -> ReLU -> Dropout -> Linear (softmax in the loss)."""

    def __init__(self, d_in, n_classes, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout / 2), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, x):
        return x + self.f(x)


class ResMLP(nn.Module):
    """Deeper MLP: projection, `layers` pre-LayerNorm residual blocks, LayerNorm, linear head."""

    def __init__(self, d_in, n_classes, hidden, dropout, layers):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout / 2), nn.Linear(d_in, hidden))
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.out(self.blocks(self.inp(x)))


class TorchClassifier:
    """sklearn-style wrapper: standardises inputs, trains with AdamW, early stopping on a validation set."""

    def __init__(self, arch="mlp", hidden=512, dropout=0.3, lr=1e-3, weight_decay=1e-4, layers=2,
                 batch_size=64, max_epochs=200, patience=15, epochs=None, device="cpu", seed=SEED):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def _prep(self, X, fit=False):
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        X = X.astype(np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        return torch.from_numpy((X - self.mu_) / self.sd_)

    def _build(self, d_in, n_classes):
        if self.arch == "mlp":
            return MLP(d_in, n_classes, self.hidden, self.dropout)
        return ResMLP(d_in, n_classes, self.hidden, self.dropout, self.layers)

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        Xt, yt = self._prep(X, fit=True), torch.as_tensor(y, dtype=torch.long)
        self.n_classes_ = int(max(y.max(), (y_val.max() if y_val is not None else 0))) + 1
        model = self._build(Xt.shape[1], self.n_classes_).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.CrossEntropyLoss()
        Xv = self._prep(X_val).to(self.device) if X_val is not None else None
        n_epochs = self.epochs or self.max_epochs
        best, best_state, wait, self.best_epoch_ = -np.inf, None, 0, n_epochs
        for ep in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(len(yt), generator=g)
            for i in range(0, len(yt), self.batch_size):
                b = perm[i : i + self.batch_size]
                opt.zero_grad()
                lossf(model(Xt[b].to(self.device)), yt[b].to(self.device)).backward()
                opt.step()
            if Xv is not None and self.epochs is None:
                model.eval()
                with torch.no_grad():
                    pred = model(Xv).argmax(1).cpu().numpy()
                val = balanced_accuracy_score(y_val, pred)
                if val > best:
                    best, wait, self.best_epoch_ = val, 0, ep
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_, self.val_score_ = model, best
        return self

    def predict_proba(self, X):
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(self._prep(X).to(self.device)), 1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# --------------------------------------------------------------------------- model zoo + grids
def _logreg(C, sparse=False):
    return Pipeline([("scale", StandardScaler(with_mean=not sparse)), ("clf", LogisticRegression(C=C, max_iter=3000))])


def _logreg_ovr(C, sparse=False):
    return Pipeline([("scale", StandardScaler(with_mean=not sparse)), ("clf", OneVsRestClassifier(LogisticRegression(C=C, max_iter=3000)))])


@dataclass
class ModelSpec:
    name: str
    build: callable  # (params, sparse) -> estimator
    grid: dict
    neural: bool = False
    params: list = field(init=False)

    def __post_init__(self):
        keys = list(self.grid)
        self.params = [dict(zip(keys, v)) for v in itertools.product(*self.grid.values())]


MODELS = {
    "logreg": ModelSpec("logreg", lambda p, s: _logreg(p["C"], s), {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}),
    "logreg_ovr": ModelSpec("logreg_ovr", lambda p, s: _logreg_ovr(p["C"], s), {"C": [1e-2]}),
    "mlp": ModelSpec(
        "mlp",
        lambda p, s: TorchClassifier("mlp", **p),
        {"hidden": [256, 1024], "dropout": [0.2, 0.5], "lr": [1e-3, 3e-4], "weight_decay": [1e-4, 1e-2]},
        neural=True,
    ),
    "resmlp": ModelSpec(
        "resmlp",
        lambda p, s: TorchClassifier("resmlp", **p),
        {"layers": [2, 3], "hidden": [256, 512], "dropout": [0.3, 0.5], "lr": [1e-3, 3e-4], "weight_decay": [1e-2]},
        neural=True,
    ),
}


# --------------------------------------------------------------------------- CV loop
def _spec(model: str, grid: dict | None) -> ModelSpec:
    return MODELS[model] if grid is None else ModelSpec(model, MODELS[model].build, grid, MODELS[model].neural)


def _fit_eval_inner(spec, params, X, y, tr, va, sparse):
    est = spec.build(params, sparse)
    if spec.neural:
        est.fit(X[tr], y[tr], X[va], y[va])
        return est.val_score_, est.best_epoch_
    est.fit(X[tr], y[tr])
    return balanced_accuracy_score(y[va], est.predict(X[va])), None


def fit_tuned(spec, X, y, groups, tr, seed: int = SEED):
    """Nested tuning inside the index set `tr`, then refit on all of `tr`. Returns (estimator, info)."""
    sparse = sp.issparse(X)
    itr, iva = inner_split(tr, groups, seed=seed)
    if len(spec.params) > 1 or spec.neural:
        res = [(_fit_eval_inner(spec, p, X, y, itr, iva, sparse), p) for p in spec.params]
        (best_val, best_ep), best_p = max(res, key=lambda t: t[0][0])
    else:
        best_p, best_val, best_ep = spec.params[0], np.nan, None
    est = spec.build(best_p, sparse)
    if spec.neural:
        est.epochs = best_ep  # refit on all of `tr` for the selected number of epochs
    est.fit(X[tr], y[tr])
    return est, {"inner_val": best_val, "params": best_p, "epochs": best_ep}


def _one_fold(spec, X, y, groups, r, k, tr, te):
    torch.set_num_threads(1)
    set_seed(SEED + r)
    est, info = fit_tuned(spec, X, y, groups, tr, seed=SEED + r)
    pred = est.predict(X[te])
    proba = est.predict_proba(X[te]) if hasattr(est, "predict_proba") else None
    row = {"repeat": r, "fold": k, **scores(y[te], pred), **info, "n_test": len(te), "n_total": len(tr) + len(te)}
    return row, te, pred, proba


def run_cv(X, y, groups, folds, model: str, grid: dict | None = None, name: str | None = None,
           n_jobs: int = -1, verbose: bool = True) -> dict:
    """Nested CV of one model on one representation; folds run in parallel (n_jobs).

    folds: output of load_folds. Returns per-fold metrics, chosen params and out-of-fold predictions
    for repeat 0 (so all analyses use the same predictions). Results are deterministic for any n_jobs.
    """
    from joblib import Parallel, delayed

    spec = _spec(model, grid)
    y, groups = np.asarray(y), np.asarray(groups)
    X = X.tocsr() if sp.issparse(X) else np.asarray(X)
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(_one_fold)(spec, X, y, groups, *f) for f in folds)
    rows = [o[0] for o in out]
    oof_pred, oof_proba = np.full(len(y), -1), np.zeros((len(y), int(y.max()) + 1))
    for row, te, pred, proba in out:
        if row["repeat"] == 0:
            oof_pred[te] = pred
            if proba is not None:
                oof_proba[te] = proba
    secs = time.time() - t0
    res = {"name": name or model, "model": model, "folds": rows, "oof_pred": oof_pred, "oof_proba": oof_proba,
           "seconds": secs, "runs": len({f[0] for f in folds})}
    if verbose:
        ba = fold_scores(res)
        print(f"{res['name']:28s} bal.acc {ba.mean():.4f} ± {ba.std():.4f}  ({len(rows)} folds, {secs:.0f}s)")
    return res


def final_test(X_train, y, groups, X_test, y_test, model: str, grid: dict | None = None) -> dict:
    """The single held-out evaluation: tune on an inner grouped split of the full train split, refit, score test."""
    set_seed(SEED)
    spec = _spec(model, grid)
    X_train = X_train.tocsr() if sp.issparse(X_train) else np.asarray(X_train)
    est, info = fit_tuned(spec, X_train, np.asarray(y), np.asarray(groups), np.arange(len(y)))
    pred = est.predict(X_test)
    return {"model": model, **scores(y_test, pred), **info, "pred": pred}


def subset_repeats(result: dict, repeats: int) -> dict:
    """Restrict a run_cv result to its first `repeats` repeats (to pair it with a costlier model)."""
    return {**result, "folds": [f for f in result["folds"] if f["repeat"] < repeats], "runs": repeats}


# --------------------------------------------------------------------------- persistence
RESULTS_DIR = Path(__file__).with_name("results")


def save_result(result: dict, path: Path | None = None) -> Path:
    """Save a run_cv result as JSON (shared/results/<name>.json) so teammates can compare against it."""
    path = Path(path or RESULTS_DIR / f"{result['name'].replace('/', '__')}.json")
    path.parent.mkdir(exist_ok=True)
    result = {**result, "oof_proba": np.round(np.asarray(result["oof_proba"]), 4)}
    conv = lambda o: o.tolist() if isinstance(o, np.ndarray) else (o.item() if isinstance(o, np.generic) else str(o))
    path.write_text(json.dumps(result, default=conv))
    return path


def load_results(pattern: str = "*.json", path: Path = RESULTS_DIR) -> dict:
    out = {}
    for f in sorted(Path(path).glob(pattern)):
        r = json.loads(f.read_text())
        r["oof_pred"], r["oof_proba"] = np.array(r["oof_pred"]), np.array(r["oof_proba"])
        out[r["name"]] = r
    return out


def fold_scores(result: dict, metric: str = "balanced_accuracy") -> np.ndarray:
    return np.array([f[metric] for f in result["folds"]])


# --------------------------------------------------------------------------- Bayesian comparison
def correlated_ttest(a, b, rho: float, rope: float = ROPE) -> dict:
    """Bayesian correlated t-test on paired per-fold scores (B - A).

    Posterior of the mean difference is Student-t with n-1 df, location mean(d) and
    scale^2 = var(d) * (1/n + rho/(1-rho)) (Nadeau-Bengio correction, Benavoli et al. 2017).
    For k-fold CV rho = 1/k, which is what baycomp hard-codes; we pass the actual n_test/n_total.
    """
    d = np.asarray(b, float) - np.asarray(a, float)
    n = len(d)
    mean, var = d.mean(), d.var(ddof=1) * (1 / n + rho / (1 - rho))
    sd = np.sqrt(var) if var > 0 else 1e-12
    t = st.t(n - 1, loc=mean, scale=sd)
    return {"mean_diff": mean, "sd": sd, "df": n - 1,
            "p_a_better": t.cdf(-rope), "p_rope": t.cdf(rope) - t.cdf(-rope), "p_b_better": 1 - t.cdf(rope)}


def compare(res_a: dict, res_b: dict, metric: str = "balanced_accuracy", rope: float = ROPE) -> dict:
    """Compare two run_cv results evaluated on the same folds."""
    fa, fb = res_a["folds"], res_b["folds"]
    if len(fa) != len(fb):  # pair on the shared repeats (e.g. 10-repeat LR vs 1-repeat fine-tuning)
        n = min(res_a["runs"], res_b["runs"])
        res_a, res_b = subset_repeats(res_a, n), subset_repeats(res_b, n)
        fa, fb = res_a["folds"], res_b["folds"]
    assert [(f["repeat"], f["fold"]) for f in fa] == [(f["repeat"], f["fold"]) for f in fb], "different folds"
    rho = np.mean([f["n_test"] / f["n_total"] for f in fa])
    out = correlated_ttest(fold_scores(res_a, metric), fold_scores(res_b, metric), rho, rope)
    return {"A": res_a.get("name", res_a["model"]), "B": res_b.get("name", res_b["model"]), "rho": rho,
            "n_folds": len(fa), **out}


def timed(fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, time.time() - t0
