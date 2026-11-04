"""Shared analysis for NLP1 Project 1: identical tables and figures for every representation and model.

    from shared import analysis as A
    A.results_table(RESULTS)                          # mean ± sd per system
    A.comparison_table(RESULTS, baseline="emb/logreg", pairs=[("emb/mlp", "emb/resmlp")])
    A.plot_posteriors(cmps)                           # Bayesian posteriors with the ROPE shaded
    A.plot_confusion(D.y[mask], res["oof_pred"][mask], D)   # rows/cols ordered by family
    A.family_errors(...); A.per_class_f1(...); A.accuracy_by_length(...); A.misclassified(...)

`RESULTS` is a dict name -> eval_harness.run_cv result (or eval_harness.load_results()).
Out-of-fold predictions come from CV repeat 0, so every row of the train split has exactly one.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import confusion_matrix, f1_score, recall_score

from shared import eval_harness as H
from shared.data import FAMILIES, FAMILY_COLOR, family

METRICS = ["balanced_accuracy", "accuracy", "macro_f1"]


def results_table(results: dict, metrics=METRICS) -> pd.DataFrame:
    """Mean ± sd over folds per system, sorted by the primary metric."""
    rows = []
    for name, r in results.items():
        f = pd.DataFrame(r["folds"])
        row = {"system": name, "folds": len(f), "minutes": r.get("seconds", np.nan) / 60}
        for m in metrics:
            row[m] = f"{f[m].mean():.4f} ± {f[m].std():.4f}"
            row[f"_{m}"] = f[m].mean()
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(f"_{metrics[0]}", ascending=False).set_index("system")
    return t.drop(columns=[f"_{m}" for m in metrics]).round({"minutes": 1})


def comparison_table(results: dict, baseline: str | None = None, pairs=(), rope: float = H.ROPE,
                     metric: str = "balanced_accuracy") -> tuple[pd.DataFrame, list]:
    """Bayesian correlated t-tests: every system vs `baseline`, plus the explicit (A, B) `pairs`.

    Columns: P(A>B), P(ROPE), P(B>A) and the posterior mean of B - A. Returns (table, raw comparisons).
    """
    todo = [(baseline, n) for n in results if baseline and n != baseline] + list(pairs)
    cmps = [H.compare(results[a], results[b], metric=metric, rope=rope) for a, b in todo]
    t = pd.DataFrame([{"A": c["A"], "B": c["B"], "folds": c["n_folds"], "mean B−A": c["mean_diff"],
                       "P(A>B)": c["p_a_better"], "P(ROPE)": c["p_rope"], "P(B>A)": c["p_b_better"]} for c in cmps])
    return t.round(4), cmps


def plot_posteriors(cmps: list, rope: float = H.ROPE, ncols: int = 3, path=None):
    """One panel per comparison: Student-t posterior of B − A (percentage points), ROPE shaded."""
    n = len(cmps)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.4 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cmps):
        t = st.t(c["df"], loc=c["mean_diff"], scale=c["sd"])
        lo, hi = min(t.ppf(0.001), -2 * rope), max(t.ppf(0.999), 2 * rope)
        x = np.linspace(lo, hi, 400)
        ax.plot(100 * x, t.pdf(x), color="#2a78d6", lw=1.5)
        ax.fill_between(100 * x, t.pdf(x), where=np.abs(x) <= rope, color="#888888", alpha=0.35, lw=0)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(f"{c['B']}\nvs {c['A']}", fontsize=8)
        ax.text(0.02, 0.95, f"P(B>A)={c['p_b_better']:.2f}\nP(ROPE)={c['p_rope']:.2f}\nP(A>B)={c['p_a_better']:.2f}",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.set_yticks([])
        ax.set_xlabel("B − A, balanced accuracy (pp)", fontsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_confusion(y_true, y_pred, D, title: str = "", path=None):
    """Row-normalised confusion matrix, classes ordered by family, family blocks outlined."""
    order = [D.class_id[c] for c in D.class_order]
    cm = confusion_matrix(y_true, y_pred, labels=order, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(order)), D.class_order, rotation=70, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), D.class_order, fontsize=7)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(FAMILY_COLOR[family(lbl.get_text())])
    start = 0
    for fam in FAMILIES:  # outline the within-family blocks
        size = sum(family(c) == fam for c in D.class_order)
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), size, size, fill=False, ec="black", lw=1))
        start += size
    for i in range(len(order)):
        for j in range(len(order)):
            if cm[i, j] >= 0.1:
                ax.text(j, i, f"{cm[i, j]:.2f}"[1:], ha="center", va="center", fontsize=5.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    ax.grid(False)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "Confusion matrix (row-normalised)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def family_errors(y_true, y_pred, D) -> pd.Series:
    """Share of errors that stay inside the true family, plus family-level accuracy and its chance level."""
    fam_t = np.array([family(D.classes[i]) for i in y_true])
    fam_p = np.array([family(D.classes[i]) for i in y_pred])
    err = np.asarray(y_true) != np.asarray(y_pred)
    sizes = pd.Series([family(c) for c in D.classes]).value_counts()
    # If errors were uniform over the 17 wrong classes, this share would stay within the family:
    chance_within = ((sizes - 1) / (len(D.classes) - 1) * sizes / len(D.classes)).sum()
    return pd.Series({"model_accuracy": 1 - err.mean(), "family_accuracy": (fam_t == fam_p).mean(),
                      "errors": int(err.sum()), "within_family_share_of_errors": (fam_t == fam_p)[err].mean(),
                      "within_family_share_if_uniform": chance_within})


def per_class_f1(y_true, y_pred, D) -> pd.DataFrame:
    labels = list(range(len(D.classes)))
    df = pd.DataFrame({"f1": f1_score(y_true, y_pred, labels=labels, average=None),
                       "recall": recall_score(y_true, y_pred, labels=labels, average=None)}, index=D.classes)
    return df.loc[D.class_order].assign(family=lambda x: x.index.map(family))


def accuracy_by_length(y_true, y_pred, n_words, bins=(0, 75, 100, 125, 150, 1000)) -> pd.DataFrame:
    b = pd.cut(np.asarray(n_words), bins=list(bins))
    ok = np.asarray(y_true) == np.asarray(y_pred)
    return pd.DataFrame({"n": pd.Series(ok).groupby(b, observed=False).size(),
                         "accuracy": pd.Series(ok).groupby(b, observed=False).mean()})


def misclassified(df: pd.DataFrame, y_true, y_pred, D, n: int = 10, proba=None, seed: int = H.SEED) -> pd.DataFrame:
    """n misclassified passages (most confident errors first if `proba` is given, else a seeded sample)."""
    err = np.flatnonzero(np.asarray(y_true) != np.asarray(y_pred))
    if proba is not None:
        err = err[np.argsort(-np.asarray(proba)[err].max(1))][:n]
    else:
        err = np.random.default_rng(seed).choice(err, size=min(n, len(err)), replace=False)
    out = df.iloc[err][["id", "prompt_id", "text"]].copy()
    out.insert(0, "pred", [D.classes[i] for i in np.asarray(y_pred)[err]])
    out.insert(0, "true", [D.classes[i] for i in np.asarray(y_true)[err]])
    if proba is not None:
        out["confidence"] = np.asarray(proba)[err].max(1).round(3)
    return out
