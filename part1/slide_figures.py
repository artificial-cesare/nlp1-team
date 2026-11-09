"""Slide figures for Project 1, built from the saved results in shared/results/ (nothing is retrained).

    .venv/bin/python part1/slide_figures.py

Writes PNGs (and a CSV per table) to part1/figures/slides/:
  1_hyperparameters      every hyperparameter tuned per model, its search space, and what was fixed
  2_model_comparison     LR / MLP / ResMLP on token indicators, bag-of-words, character n-grams and embeddings
  3_transformers         from-scratch transformer and the fine-tuned LUSR variants
  4_bayes_rope           Bayesian correlated t-test for a subset of comparisons, ROPE = 1 pp
  5_confusion_best       out-of-fold confusion matrix of the best system (prompt-centred Qwen3-Base + LR)
  1b, 2b                 shorter slide versions of tables 1 and 2
  6_cv_vs_test           CV vs held-out test vs a Kaggle-like test file (prompts recovered by clustering)
  7_protocol             diagram of the nested, prompt-grouped CV protocol
  8_best_result          how the score moved from the length baseline to the best system
"""

from __future__ import annotations

import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as st  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402

from shared import eval_harness as H  # noqa: E402
from shared.data import FAMILIES, FAMILY_COLOR, family, load_data  # noqa: E402

OUT = ROOT / "part1" / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
R = H.load_results()
D = load_data()

# Chart ink and the sequential blue ramp (dataviz reference palette)
INK, INK2, MUTED, RULE, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#c3c2b7"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": BASE, "savefig.facecolor": "white"})


def ba(name, repeats=None):
    r = R[name] if repeats is None else H.subset_repeats(R[name], repeats)
    return H.fold_scores(r)


def fmt(name, repeats=None, sd=True):
    s = ba(name, repeats)
    return f"{100 * s.mean():.1f}" + (f" ± {100 * s.std(ddof=1):.1f}" if sd else "")


def modal(name, key="params", n=1):
    c = Counter(json.dumps(f[key], sort_keys=True) for f in R[name]["folds"])
    return c.most_common(n)


# --------------------------------------------------------------------------- generic table renderer
def draw_table(header, rows, widths, path, title, subtitle="", footnote="", sections=None, fills=None,
               bold_rows=(), font=9.0, header_font=9.0, outline=(), bold_cells=()):
    """Render a text table as a figure. widths: column widths in inches. sections: {row index: label} printed
    as a band above that row. fills: {(row, col): (background, text colour)}."""
    sections, fills = sections or {}, fills or {}
    chars_per_inch = 13.6 * 9 / font
    wrap = lambda s, w, k=1.0: textwrap.wrap(str(s), max(4, int((w - 0.14) * chars_per_inch * k))) or [""]  # noqa: E731
    line_h, pad = font / 72 * 1.35, 0.07
    hdr_lines = [wrap(h, w, 0.88) for h, w in zip(header, widths)]
    hdr_h = max(map(len, hdr_lines)) * line_h + 2 * pad
    body = [[wrap(c, w, 0.9 if i in bold_rows else 1.0) for c, w in zip(row, widths)] for i, row in enumerate(rows)]
    row_h = [max(map(len, r)) * line_h + 2 * pad for r in body]
    sec_h = line_h + 2 * pad
    title_h = 0.45 + (0.25 if subtitle else 0)
    foot_lines = sum(len(wrap(p, sum(widths) * 1.04)) for p in footnote.split("\n")) if footnote else 0
    foot_h = foot_lines * line_h * 0.95 + (0.15 if footnote else 0)
    W, Hh = sum(widths) + 0.3, title_h + hdr_h + sum(row_h) + sec_h * len(sections) + foot_h + 0.2
    fig = plt.figure(figsize=(W, Hh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(Hh, 0)
    ax.axis("off")
    x0 = 0.15
    ax.text(x0, 0.3, title, fontsize=13, weight="bold", va="center")
    if subtitle:
        ax.text(x0, 0.57, subtitle, fontsize=9, color=INK2, va="center")
    y = title_h
    xs = np.concatenate([[x0], x0 + np.cumsum(widths)])
    for j, lines in enumerate(hdr_lines):
        for k, ln in enumerate(lines):
            ax.text(xs[j] + 0.06, y + pad + (k + 0.5) * line_h, ln, fontsize=header_font, weight="bold", color=INK2,
                    va="center")
    y += hdr_h
    ax.plot([x0, xs[-1]], [y, y], color=INK, lw=1.0)
    for i, (cells, h) in enumerate(zip(body, row_h)):
        if i in sections:
            ax.text(x0 + 0.06, y + sec_h / 2 + 0.02, sections[i], fontsize=font, weight="bold", color=BLUE,
                    va="center")
            y += sec_h
        for j, lines in enumerate(cells):
            bg, fg = fills.get((i, j), (None, INK))
            if bg:
                ax.add_patch(plt.Rectangle((xs[j] + 0.02, y + 0.02), widths[j] - 0.04, h - 0.04, color=bg, lw=0))
            if (i, j) in outline:
                ax.add_patch(plt.Rectangle((xs[j] + 0.02, y + 0.02), widths[j] - 0.04, h - 0.04, fill=False,
                                           ec=INK, lw=2.4, zorder=5))
            for k, ln in enumerate(lines):
                ax.text(xs[j] + 0.06, y + pad + (k + 0.5) * line_h, ln, fontsize=font, color=fg, va="center",
                        weight="bold" if i in bold_rows or (i, j) in bold_cells else "normal")
        y += h
        ax.plot([x0, xs[-1]], [y, y], color=RULE, lw=0.6)
    ax.plot([x0, xs[-1]], [y, y], color=INK, lw=1.0)
    if footnote:
        y += 0.12
        for para in footnote.split("\n"):
            for ln in wrap(para, sum(widths) * 1.04):
                ax.text(x0, y + line_h * 0.5, ln, fontsize=font - 1, color=INK2, va="center")
                y += line_h * 0.95
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print("wrote", path.relative_to(ROOT))


def seq_fill(v, lo=10, hi=50):
    """Sequential blue background for a score in %, with white text on the dark half."""
    i = int(np.clip((v - lo) / (hi - lo), 0, 1) * (len(BLUE_RAMP) - 1))
    return BLUE_RAMP[i], ("white" if i >= 6 else INK)


# --------------------------------------------------------------------------- 1 · hyperparameters
def table_hyperparameters():
    lr_emb = ", ".join(f"{json.loads(k)['C']:g} ({v})" for k, v in modal("lusr/logreg", n=2))
    lr_tok = ", ".join(f"{json.loads(k)['C']:g} ({v})" for k, v in modal("bow-ours2k/logreg_scale_tuned", n=2))
    mlp = json.loads(modal("lusr/mlp")[0][0])
    res = json.loads(modal("lusr/resmlp")[0][0])
    no_scale_bow = sum(not f["params"]["scale"] for f in R["bow-ours2k/logreg_scale_tuned"]["folds"])
    no_scale_ind = sum(not f["params"]["scale"] for f in R["indicator-ours2k/logreg_scale_tuned"]["folds"])
    # gain from switching standardisation off on the count features (outer folds; MLPs on their 5 common folds)
    count_reps = [k.split("/")[0] for k in R if k.endswith("/logreg_unscaled")]
    gain = lambda m: [100 * H.compare(R[f"{r}/{m}-sdfix" if m != "logreg" else f"{r}/{m}"],  # noqa: E731
                                      R[f"{r}/{m}_unscaled"])["mean_diff"]
                      for r in count_reps if f"{r}/{m}_unscaled" in R and (m == "logreg" or f"{r}/{m}-sdfix" in R)]
    g_lr, g_mlp, g_res = gain("logreg"), gain("mlp"), gain("resmlp")
    ep = lambda n: ", ".join(str(f["epochs"]) for f in R[n]["folds"])  # noqa: E731
    d_tr = Counter(f["params"]["d_model"] for f in R["roberta-ids/transformer"]["folds"])
    header = ["Model", "Tuned on the inner validation split", "Settings", "Fixed (not tuned)",
              "Most often chosen (folds)"]
    rows = [
        ["Logistic regression (multinomial)",
         "standardise input {yes, no} × C (inverse L2 strength) ∈ {1e-4, 1e-3, …, 1e4} (9 values)", "18",
         "L-BFGS, max 3,000 iterations",
         f"count features: no scaling in {no_scale_bow}/50 folds (BoW 2k) and {no_scale_ind}/50 (indicators 2k); "
         f"BoW 2k: C = {lr_tok} of 50. Embeddings: standardised, LUSR C = {lr_emb} of 50"],
        ["Logistic regression, one-vs-rest", "nothing", "1", "C = 0.01, standardised (cheap check)", "–"],
        ["MLP, 1 hidden layer", "standardise input {yes, no} × width {256, 1024} × dropout {0.2, 0.5} × learning "
         "rate {1e-3, 3e-4} × weight decay {1e-4, 1e-2}; + number of epochs (early stopping)", "2 × 16",
         "ReLU, input dropout = dropout/2, AdamW, batch 64, ≤ 200 epochs, patience 15",
         f"count features: no scaling (+{min(g_mlp):.1f} to +{max(g_mlp):.1f} pp). LUSR: standardised, width "
         f"{mlp['hidden']}, dropout {mlp['dropout']}, lr {mlp['lr']:g}"],
        ["Residual MLP", "standardise input {yes, no} × blocks {2, 3} × width {256, 512} × dropout {0.3, 0.5} × "
         "learning rate {1e-3, 3e-4}; + number of epochs (early stopping)", "2 × 16",
         "pre-LayerNorm blocks, GELU, weight decay 1e-2, AdamW, batch 64, patience 15",
         f"count features: no scaling ({abs(min(g_res)) if abs(min(g_res)) < 0.05 else min(g_res):+.1f} to +{max(g_res):.1f} pp). LUSR: standardised, "
         f"{res['layers']} blocks, width {res['hidden']}, dropout {res['dropout']}, lr {res['lr']:g}"],
        ["Transformer from scratch", "d_model {128, 256}; + number of epochs (early stopping)", "2",
         "2 layers, 4 heads, FF 4·d, dropout 0.1, lr 5e-4, weight decay 0.01, batch 32, ≤ 60 epochs, patience 8; "
         "vocabulary = RoBERTa ids seen ≥ 2× in the training part",
         f"d_model = 256 in {d_tr[256]}/5 folds; epochs {ep('roberta-ids/transformer')}"],
        ["LUSR fine-tuned, all layers", "epoch: best of the 5 (or 10) end-of-epoch checkpoints", "5 / 10",
         "lr 2e-5 (head 1e-3), weight decay 0.01, 10% linear warm-up then linear decay to 0, batch 16, "
         "mean pooling", f"epochs {ep('lusr-finetune/finetune')} (5-epoch run)"],
        ["LUSR fine-tuned, top 2 / top 4 layers", "epoch: best of 6 checkpoints; with refit, retrain on the whole "
         "training fold for that many epochs", "6",
         "as above but lr 5e-5, head dropout 0.1, lower layers and embeddings frozen",
         f"top 4: epochs {ep('lusr-finetune-top4-refit/finetune')}"],
        ["Prompt recovery for centring (Kaggle)", "clustering distance threshold, by adjusted Rand index on the "
         "train split", "158 (1 threshold)", "average linkage, cosine distance on Qwen3-Embedding vectors",
         "threshold 0.385 (ARI 0.988 on the test split)"],
    ]
    foot = ("Nested CV: the setting is chosen on an inner 20% of each training fold (split by prompt), refit on the "
            "whole training fold and scored once on the outer fold, so the reported scores are not used to tune.\n"
            f"Standardisation: the inner split switched it off for logistic regression on the 2k count features, so "
            f"it was switched off for every count feature set and model ({min(g_lr):+.1f} to {max(g_lr):+.1f} pp "
            f"for logistic regression on the outer folds); dense embeddings and the length baseline stay "
            f"standardised, with C ∈ {{1e-4 … 10}}.\n"
            "Chosen by comparing systems on the outer folds, not tuned per fold: tokeniser vocabulary, Qwen "
            "layer (14, fixed in advance), pooling (mean vs mean‖max), fine-tuning variant (top 4), ROPE (1 pp).")
    draw_table(header, rows, [1.9, 3.0, 0.9, 3.0, 2.6], OUT / "1_hyperparameters.png",
               "Hyperparameters tuned per model", "nested cross-validation, 50 folds (5 for the slow models)",
               footnote=foot, font=8.5)
    pd.DataFrame(rows, columns=header).to_csv(OUT / "1_hyperparameters.csv", index=False)
    # slide version: the rubric architectures plus the two end-to-end models, without the "fixed" column
    keep_rows, keep_cols = [0, 2, 3, 4, 6], [0, 1, 2, 4]
    short = [[rows[i][j] for j in keep_cols] for i in keep_rows]
    draw_table([header[j] for j in keep_cols], short, [1.9, 3.4, 1.05, 3.0], OUT / "1b_hyperparameters_slide.png",
               "Hyperparameters tuned per model", "chosen on an inner validation split inside every CV fold "
               "(nested), then refit on the whole training fold", font=9,
               footnote="Standardisation is switched off for count features and kept for dense embeddings. "
                        "Fixed settings and the extra models: 1_hyperparameters.png.")


# --------------------------------------------------------------------------- 2 · model comparison
def table_models(reps=None, stem="2_model_comparison", slide=False):
    reps = reps or [
        ("Token indicators", [("indicator-ours2k", "Our BPE, 2k vocabulary"), ("indicator-ours4k", "Our BPE, 4k"),
                                     ("indicator-ours8k", "Our BPE, 8k"), ("indicator-lusr", "LUSR's BPE, 50k")]),
        ("Bag-of-words", [("bow-ours2k", "Our BPE, 2k vocabulary"), ("bow-ours4k", "Our BPE, 4k"),
                                 ("bow-ours8k", "Our BPE, 8k"), ("bow-lusr", "LUSR's BPE, 50k")]),
        ("Other features", [("char-2-4grams", "Character 2–4-grams (20k)"), ("length", "Length only (baseline)")]),
        ("Frozen embeddings", [("lusr", "LUSR, style (768-d)"), ("qwen3-base-L14-mean", "Qwen3-0.6B-Base, layer 14 (1024-d)"),
                               ("qwen3-base-L14-meanmax", "Qwen3-Base L14, mean‖max (2048-d)"),
                               ("qwen3-emb", "Qwen3-Embedding, semantic (1024-d)")]),
        ("Learned end to end (linear softmax head)", [
            ("roberta-ids", "Transformer from scratch (RoBERTa ids)", "roberta-ids/transformer"),
            ("lusr-ft", "LUSR fine-tuned, all layers, 5 epochs", "lusr-finetune/finetune"),
            ("lusr-ft10", "LUSR fine-tuned, all layers, 10 epochs", "lusr-finetune-10ep/finetune"),
            ("lusr-ft2", "LUSR fine-tuned, top 2 layers", "lusr-finetune-top2/finetune"),
            ("lusr-ft4", "LUSR fine-tuned, top 4 layers", "lusr-finetune-top4/finetune"),
            ("lusr-ft4r", "LUSR fine-tuned, top 4 layers + refit", "lusr-finetune-top4-refit/finetune")]),
        ("Prompt-centred embeddings", [("lusr-promptcentred", "LUSR"), ("qwen3-base-L14-mean-promptcentred",
                                                                         "Qwen3-Base, layer 14"),
                                       ("qwen3-emb-promptcentred", "Qwen3-Embedding")]),
    ]
    header = ["Representation", "Logistic regression / linear head", "MLP", "Residual MLP"]
    rows, sections, fills, csv, cells = [], {}, {}, [], {}
    for sec, items in reps:
        sections[len(rows)] = sec
        for key, label, *end_to_end in items:
            row, rec = [label], {"group": sec, "representation": label}
            for j, m in enumerate(["logreg", "mlp", "resmlp"], start=1):
                if end_to_end:
                    name = end_to_end[0] if j == 1 else None
                else:
                    # count features: unscaled runs; embeddings and length: standardised (-sdfix for MLPs)
                    name = next((n for n in (f"{key}/{m}_unscaled", f"{key}/{m}-sdfix", f"{key}/{m}") if n in R),
                                None)
                if name is None:
                    row.append("–" if end_to_end else "not run")
                    fills[(len(rows), j)] = (None, MUTED)
                    rec[m] = np.nan
                    continue
                cells[(len(rows), j)] = ba(name).mean()
                s = ba(name)
                dag = " †" if len(s) == 5 else ""
                row.append(f"{100 * s.mean():.1f} ± {100 * s.std(ddof=1):.1f}{dag}")
                fills[(len(rows), j)] = seq_fill(100 * s.mean())
                rec[m], rec[f"{m}_folds"] = round(100 * s.mean(), 2), len(s)
            rows.append(row)
            csv.append(rec)
    bi, bj = max(cells, key=cells.get)  # best cell overall
    rows[bi][bj] += "  ★ best"
    rows[bi][0] += "  ★"
    fills[(bi, bj)] = ("#104281", "white")
    foot = ("Balanced accuracy in % on the train split, mean ± sd over folds (chance 5.6%). 50 folds unless marked "
            "† = first 5 folds only (token MLPs: 1–2.5 h per 50-fold run; transformer and fine-tuning: 7–55 min per "
            "5 folds). The end-to-end models finish with a linear softmax layer, so they sit in the first column. "
            "★ = best system: prompt-centred Qwen3-Base + logistic regression (47.6% on the held-out test split with "
            "prompts recovered by clustering).\n"
            "Count features (indicators, bag-of-words, character n-grams) are not standardised: the inner split "
            "switched standardisation off, and dropping it gains 1–13 pp for logistic regression. Embeddings and "
            "the length baseline are standardised.")
    if slide:
        foot = ("Balanced accuracy (%), mean ± sd over 50 CV folds; † = first 5 folds only. Chance 5.6%. "
                "★ best system (47.6% on the held-out test split, prompts recovered by clustering). "
                "Count features unscaled, embeddings standardised. "
                "All rows: 2_model_comparison.png.")
    draw_table(header, rows, [3.1, 2.0, 1.5, 1.5], OUT / f"{stem}.png",
               "Models and representations", "cross-validated balanced accuracy (%), grouped by prompt",
               footnote=foot, sections=sections, fills=fills, outline={(bi, bj)}, bold_cells={(bi, 0), (bi, bj)})
    pd.DataFrame(csv).to_csv(OUT / f"{stem}.csv", index=False)


SLIDE_REPS = [
    ("Token features (our BPE, 2k vocabulary)", [("indicator-ours2k", "Token indicators"),
                                                 ("bow-ours2k", "Bag-of-words"),
                                                 ("char-2-4grams", "Character 2–4-grams (no tokeniser)"),
                                                 ("length", "Length only (baseline)")]),
    ("Frozen embeddings", [("lusr", "LUSR, style (768-d)"), ("qwen3-base-L14-mean", "Qwen3-0.6B-Base, layer 14 (1024-d)"),
                           ("qwen3-emb", "Qwen3-Embedding, semantic (1024-d)")]),
    ("Learned end to end (linear softmax head)", [
        ("roberta-ids", "Transformer from scratch", "roberta-ids/transformer"),
        ("lusr-ft", "LUSR fine-tuned, all layers", "lusr-finetune/finetune"),
        ("lusr-ft4r", "LUSR fine-tuned, top 4 layers + refit", "lusr-finetune-top4-refit/finetune")]),
    ("Prompt-centred embeddings", [("lusr-promptcentred", "LUSR"),
                                   ("qwen3-base-L14-mean-promptcentred", "Qwen3-Base, layer 14")]),
]


# --------------------------------------------------------------------------- 3 · transformers
def scratch_params():
    """Parameter count of the from-scratch transformer, vocabulary averaged over the 5 inner-training parts."""
    from shared.text_models import TinyTransformer
    from shared.tokenization import tokenize

    d = D.train.reset_index(drop=True)
    ids = tokenize(d["text"].tolist(), "roberta-bpe")
    sizes = []
    for r, k, tr, te in H.load_folds(d["id"].tolist(), repeats=1):
        itr, _ = H.inner_split(np.asarray(tr), D.groups, seed=H.SEED + r)
        c = Counter(t for i in itr for t in ids[i])
        sizes.append(sum(v >= 2 for v in c.values()) + 2)
    m = TinyTransformer(int(np.mean(sizes)), 18, 256, 2, 4, 0.1, 512)
    return sum(p.numel() for p in m.parameters()), int(np.mean(sizes))


def lusr_params():
    from transformers import AutoModel

    enc = AutoModel.from_pretrained("rrivera1849/LUSR")
    layer = sum(p.numel() for p in enc.encoder.layer[0].parameters())
    full = sum(p.numel() for p in enc.embeddings.parameters()) + 12 * layer
    head = 768 * 18 + 18
    return {None: full + head, 4: 4 * layer + head, 2: 2 * layer + head}


def table_transformers():
    n_scratch, vocab = scratch_params()
    lp = lusr_params()
    test = json.loads((ROOT / "part1" / "figures" / "s6_finetune_refit_test.json").read_text())
    ref = "lusr/logreg"
    systems = [
        ("roberta-ids/transformer", "Transformer from scratch (2 layers, d = 256)", n_scratch, "RoBERTa BPE ids", "–"),
        ("lusr-finetune/finetune", "LUSR fine-tuned, all 12 layers, 5 epochs", lp[None], "raw text", "–"),
        ("lusr-finetune-10ep/finetune", "LUSR fine-tuned, all 12 layers, 10 epochs", lp[None], "raw text", "–"),
        ("lusr-finetune-top2/finetune", "LUSR fine-tuned, top 2 layers", lp[2], "raw text", "–"),
        ("lusr-finetune-top4/finetune", "LUSR fine-tuned, top 4 layers", lp[4], "raw text",
         f"{100 * test['no_refit']['balanced_accuracy']:.1f}"),
        ("lusr-finetune-top4-refit/finetune", "LUSR fine-tuned, top 4 layers + refit on the whole fold", lp[4],
         "raw text", f"{100 * test['refit']['balanced_accuracy']:.1f}"),
    ]
    header = ["System", "Weights trained", "Epochs chosen (per fold)", "CV balanced acc. (5 folds)",
              "vs frozen LUSR + LR: Δ (pp), P(better), P(ROPE)", "Test split", "Minutes (5 folds)"]
    rows, fills, csv = [], {}, []
    for name, label, n, _, t in systems:
        c = H.compare(R[ref], R[name])
        eps = ", ".join(str(f["epochs"]) for f in R[name]["folds"])
        rows.append([label, f"{n / 1e6:.1f}M", eps, fmt(name),
                     f"{100 * c['mean_diff']:+.1f}  ·  {c['p_b_better']:.2f}  ·  {c['p_rope']:.2f}", t,
                     f"{R[name]['seconds'] / 60:.0f}"])
        fills[(len(rows) - 1, 3)] = seq_fill(100 * ba(name).mean())
        csv.append({"system": label, "trained_params": n, "epochs": eps, "cv_mean": 100 * ba(name).mean(),
                    "cv_sd": 100 * ba(name).std(ddof=1), "diff_vs_frozen_lr_pp": 100 * c["mean_diff"],
                    "p_better": c["p_b_better"], "p_rope": c["p_rope"], "test": t,
                    "minutes": R[name]["seconds"] / 60})
    test_tab = pd.read_csv(ROOT / "part1" / "figures" / "s7b_centring_test_table.csv", index_col=0)
    for name, label, tname in [(ref, "Reference: frozen LUSR embedding + LR", "lusr/logreg"),
                               ("lusr-promptcentred/logreg", "Reference: prompt-centred LUSR + LR",
                                "lusr-promptcentred/logreg")]:
        rows.append([label, "18 × 768 (LR)", "–", fmt(name, repeats=1), "–",
                     f"{100 * test_tab.loc[tname, 'balanced_accuracy']:.1f}", "< 1"])
        fills[(len(rows) - 1, 3)] = seq_fill(100 * ba(name, 1).mean())
    foot = (f"Same first 5 CV folds for every row. The from-scratch vocabulary is the {vocab:,} RoBERTa ids seen at "
            "least twice in the inner-training part. The epoch is chosen on the inner validation split (no refit, "
            "except the last fine-tuning row). Δ and probabilities: Bayesian correlated t-test against frozen LUSR + "
            "LR on the same 5 folds, ROPE = 1 pp. Test split: 540 held-out passages, scored once (± ~2 pp). The "
            "top-4 variant was picked by comparing variants on these folds, so its CV score is slightly optimistic.")
    draw_table(header, rows, [3.2, 1.05, 1.35, 1.35, 2.25, 0.8, 1.0], OUT / "3_transformers.png",
               "Transformers: trained from scratch vs fine-tuned", "RoBERTa-base LUSR encoder (125M parameters), "
               "mean pooling + linear head; MPS laptop GPU", footnote=foot, fills=fills, bold_rows=(5,))
    pd.DataFrame(csv).to_csv(OUT / "3_transformers.csv", index=False)


# --------------------------------------------------------------------------- 4 · Bayesian comparison
def figure_bayes():
    groups = [
        ("Representation", [
            ("indicator-ours8k/logreg_unscaled", "bow-ours8k/logreg_unscaled", "Indicators vs bag-of-words (8k BPE, LR)"),
            ("bow-lusr/logreg_unscaled", "lusr/logreg", "Best bag-of-words (LUSR tokeniser) vs LUSR embedding (LR)"),
            ("lusr/logreg", "qwen3-base-L14-mean/logreg", "LUSR vs Qwen3-Base L14 (LR)"),
            ("qwen3-base-L14-mean/logreg", "qwen3-emb/logreg", "Qwen3-Base vs Qwen3-Embedding (LR)"),
        ]),
        ("Architecture (LUSR)", [
            ("lusr/logreg", "lusr/mlp", "LR vs MLP"),
            ("lusr/logreg", "lusr/resmlp", "LR vs residual MLP"),
        ]),
        ("Learning the representation (5 folds)", [
            ("lusr/logreg", "roberta-ids/transformer", "Frozen LUSR + LR vs transformer from scratch"),
            ("lusr/logreg", "lusr-finetune/finetune", "Frozen LUSR + LR vs full fine-tuning"),
            ("lusr/logreg", "lusr-finetune-top4-refit/finetune", "Frozen LUSR + LR vs top-4 fine-tuning + refit"),
        ]),
        ("Prompt-centring", [
            ("lusr/logreg", "lusr-promptcentred/logreg", "Raw vs centred LUSR (LR)"),
            ("lusr-promptcentred/logreg", "qwen3-base-L14-mean-promptcentred/logreg",
             "Centred LUSR vs centred Qwen3-Base (LR)"),
        ]),
    ]
    items, csv = [], []
    for g, pairs in groups:
        for a, b, label in pairs:
            c = H.compare(R[a], R[b])
            items.append((g, label, c))
            csv.append({"group": g, "comparison (A vs B)": label, "A": a, "B": b, "folds": c["n_folds"],
                        "mean B−A (pp)": 100 * c["mean_diff"], "P(A better)": c["p_a_better"],
                        "P(ROPE)": c["p_rope"], "P(B better)": c["p_b_better"]})
    n = len(items)
    # row positions with a gap between groups
    ypos, y, prev = [], 0.0, None
    for g, _, _ in items:
        if prev is not None and g != prev:
            y += 0.9
        ypos.append(y)
        y += 1
        prev = g
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 0.42 * (ypos[-1] + 1) + 1.6),
                                 gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.04}, sharey=True)
    lim = (-21, 19)
    ax.axvspan(-1, 1, color="#e1e0d9", lw=0, zorder=0)
    ax.axvline(0, color=BASE, lw=0.8, zorder=1)
    for yy, (g, label, c) in zip(ypos, items):
        t = st.t(c["df"], loc=100 * c["mean_diff"], scale=100 * c["sd"])
        lo, hi = t.ppf(0.025), t.ppf(0.975)
        col = BLUE if c["p_b_better"] > 0.95 else ORANGE if c["p_a_better"] > 0.95 else INK2
        ax.plot([max(lo, lim[0]), min(hi, lim[1])], [yy, yy], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([np.clip(100 * c["mean_diff"], *lim)], [yy], s=42, color=col, zorder=3, ec="white", lw=1.2)
        ax.text(lim[0] - 0.4, yy, label, ha="right", va="center", fontsize=9, color=INK)
        ax.text(lim[1] - 0.2, yy, f"{100 * c['mean_diff']:+.1f}", ha="right", va="center", fontsize=8.5, color=INK2,
                bbox=dict(fc="white", ec="none", pad=1))
        # stacked probabilities
        left = 0
        for p, colr, txt in [(c["p_a_better"], ORANGE, "white"), (c["p_rope"], GREY, INK),
                             (c["p_b_better"], BLUE, "white")]:
            if p > 0.003:
                bx.barh(yy, p, left=left, height=0.62, color=colr, ec="white", lw=1.5)
                if p >= 0.12:
                    bx.text(left + p / 2, yy, f"{p:.2f}", ha="center", va="center", fontsize=8, color=txt)
            left += p
    for g in dict.fromkeys(g for g, _, _ in items):
        y0 = ypos[[i for i, it in enumerate(items) if it[0] == g][0]]
        ax.text(lim[0] - 0.4, y0 - 0.75, g, ha="right", va="center", fontsize=9.5, weight="bold", color=BLUE)
    ax.set_xlim(*lim)
    ax.set_ylim(ypos[-1] + 0.8, -1.4)
    ax.set_yticks([])
    ax.set_xlabel("B − A, balanced accuracy (percentage points); line = 95% credible interval")
    ax.text(0, -1.15, "ROPE ±1 pp", ha="center", fontsize=8, color=INK2)
    ax.text(-5, -1.15, "← A better", ha="right", fontsize=8.5, color=ORANGE, weight="bold")
    ax.text(5, -1.15, "B better →", ha="left", fontsize=8.5, color=BLUE, weight="bold")
    ax.grid(axis="x", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    bx.set_xlim(0, 1)
    bx.set_xlabel("posterior probability")
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, GREY, BLUE)]
    bx.legend(handles, ["A better", "practically equivalent (ROPE)", "B better"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8, handlelength=1, columnspacing=1)
    fig.suptitle("Bayesian correlated t-test, ROPE = 1 pp", x=0.02, ha="left", fontsize=13, weight="bold", y=0.975)
    fig.text(0.02, 0.925, "Each row: system A vs system B on the same folds (50; the first 5 for fine-tuning and the "
             "transformer). Point = posterior mean of B − A, printed on the right.\nColour: blue = B better with "
             "P > 0.95, orange = A better with P > 0.95, grey = undecided.", fontsize=9, color=INK2, va="top")
    fig.subplots_adjust(left=0.33, right=0.98, top=0.84, bottom=0.1)
    fig.savefig(OUT / "4_bayes_rope.png", dpi=220)
    plt.close(fig)
    print("wrote", (OUT / "4_bayes_rope.png").relative_to(ROOT))
    pd.DataFrame(csv).round(4).to_csv(OUT / "4_bayes_rope.csv", index=False)


# --------------------------------------------------------------------------- 5 · confusion matrix
def figure_confusion():
    name = "qwen3-base-L14-mean-promptcentred/logreg"
    y, pred = D.y, np.asarray(R[name]["oof_pred"])
    order = [D.class_id[c] for c in D.class_order]
    cm = confusion_matrix(y, pred, labels=order, normalize="true")
    fam_acc = np.mean([family(D.classes[a]) == family(D.classes[b]) for a, b in zip(y, pred)])
    fig, ax = plt.subplots(figsize=(8.6, 8.0))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#ffffff"] + BLUE_RAMP)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    labels = [c.replace("-preview", "") for c in D.class_order]
    ax.set_xticks(range(18), labels, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(18), labels, fontsize=8.5)
    for lbl, c in zip(list(ax.get_xticklabels()) + list(ax.get_yticklabels()), D.class_order * 2):
        lbl.set_color(FAMILY_COLOR[family(c)])
    start = 0
    for fam in FAMILIES:
        size = sum(family(c) == fam for c in D.class_order)
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), size, size, fill=False, ec=INK, lw=1.2))
        start += size
    for i in range(18):
        for j in range(18):
            if cm[i, j] >= 0.08:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 0.45 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    bal = balanced_accuracy_score(y, pred)
    fig.suptitle("Confusion matrix of the best system: prompt-centred Qwen3-Base L14 + LR", x=0.03, ha="left",
                 fontsize=12.5, weight="bold", y=0.985)
    fig.text(0.03, 0.945, f"Out-of-fold predictions on the 2,159 train-split passages (first CV repeat); % of each "
             f"true class (rows sum to 100; cells < 8% hidden).\nBalanced accuracy {100 * bal:.1f}% on these pooled "
             f"predictions (mean over all 50 folds: {fmt(name, sd=False)}%), family "
             f"accuracy {100 * fam_acc:.1f}%. Boxes = model families; label colour = family.", fontsize=8.5,
             color=INK2, va="top")
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.17)
    fig.savefig(OUT / "5_confusion_best.png", dpi=220)
    plt.close(fig)
    print("wrote", (OUT / "5_confusion_best.png").relative_to(ROOT))
    pd.DataFrame(cm, index=D.class_order, columns=D.class_order).round(3).to_csv(OUT / "5_confusion_best.csv")


# --------------------------------------------------------------------------- 6 · CV vs held-out test
def _test_numbers():
    fig = ROOT / "part1" / "figures"
    t7 = pd.read_csv(fig / "s7_test_table.csv", index_col=0)["balanced_accuracy"]
    tc = pd.read_csv(fig / "s7b_centring_test_table.csv", index_col=0)["balanced_accuracy"]
    k = pd.read_csv(fig / "s7b_kaggle_like_test.csv", index_col=0)
    ft = json.loads((fig / "s6_finetune_refit_test.json").read_text())
    kl = lambda emb, col: tuple(100 * k[(k["embedding"] == emb)].loc[f"subsamples: {q}", col]  # noqa: E731
                                for q in ("mean", "2.5%", "97.5%"))
    clus = lambda emb: 100 * k[(k["embedding"] == emb)].loc["full split", "centred, recovered clusters"]  # noqa: E731
    return t7, tc, kl, clus, ft


def table_cv_vs_test():
    t7, tc, kl, clus, ft = _test_numbers()
    raw, cen = "raw", "centred, recovered clusters"
    systems = [  # label, CV result, test, test with recovered clusters, Kaggle-like (mean, lo, hi)
        ("LUSR + logistic regression", "lusr/logreg", 100 * t7["lusr/logreg"], None, kl("lusr", raw)),
        ("LUSR + MLP", "lusr/mlp", 100 * t7["lusr/mlp"], None, None),
        ("LUSR + residual MLP", "lusr/resmlp", 100 * t7["lusr/resmlp"], None, None),
        ("Qwen3-Base L14 + logistic regression", "qwen3-base-L14-mean/logreg",
         100 * tc["qwen3-base-L14-mean/logreg"], None, kl("qwen3-base-L14-mean", raw)),
        ("LUSR fine-tuned, top 4 layers + refit", "lusr-finetune-top4-refit/finetune",
         100 * ft["refit"]["balanced_accuracy"], None, None),
        ("Prompt-centred LUSR + LR", "lusr-promptcentred/logreg", 100 * tc["lusr-promptcentred/logreg"],
         clus("lusr"), kl("lusr", cen)),
        ("Prompt-centred Qwen3-Base L14 + LR  ★", "qwen3-base-L14-mean-promptcentred/logreg",
         100 * tc["qwen3-base-L14-mean-promptcentred/logreg"], clus("qwen3-base-L14-mean"),
         kl("qwen3-base-L14-mean", cen)),
    ]
    header = ["System", "CV, train split", "Held-out test split", "Test, prompts recovered by clustering",
              "Kaggle-like: 50 subsamples with unequal classes (mean, 95% range)"]
    rows, fills, csv = [], {}, []
    for i, (label, name, test, cl, k) in enumerate(systems):
        s = ba(name)
        cv = f"{100 * s.mean():.1f}" + (" †" if len(s) == 5 else "")
        row = [label, cv, f"{test:.1f}", f"{cl:.1f}" if cl is not None else "no centring",
               f"{k[0]:.1f}  ({k[1]:.1f}–{k[2]:.1f})" if k is not None else "–"]
        for j, v in [(1, 100 * s.mean()), (2, test), (3, cl), (4, k[0] if k else None)]:
            fills[(i, j)] = seq_fill(v) if v is not None else (None, MUTED)
        rows.append(row)
        csv.append({"system": label, "cv": 100 * s.mean(), "cv_folds": len(s), "test": test, "test_clusters": cl,
                    "kaggle_like_mean": k[0] if k else None, "kaggle_like_lo": k[1] if k else None,
                    "kaggle_like_hi": k[2] if k else None})
    best = len(rows) - 1
    foot = ("Balanced accuracy (%). CV: 50 folds († first 5). Held-out test: 540 passages from 30 unseen prompts, "
            "used once; its standard error is about 2 pp, and every system lands within that of its CV estimate. "
            "Kaggle's test file has no prompt ids, so for the last two columns the prompts are recovered by clustering "
            "Qwen3-Embedding vectors (threshold tuned on the train split; ARI 0.988). Kaggle-like: each class keeps a "
            "random 25–100% of its passages (about 340 per subsample), so prompt groups are incomplete. "
            "For the uncentred systems the last column is the same subsamples without centring.")
    draw_table(header, rows, [3.2, 1.2, 1.3, 1.65, 2.35], OUT / "6_cv_vs_test.png",
               "Does the CV estimate hold on unseen data?", "held-out test split, and a simulation of Kaggle's test file",
               footnote=foot, fills=fills, bold_rows=(best,), outline={(best, 4)})
    pd.DataFrame(csv).round(2).to_csv(OUT / "6_cv_vs_test.csv", index=False)


# --------------------------------------------------------------------------- 7 · evaluation protocol diagram
def figure_protocol():
    TRAIN, VAL, TUNE, TEST = BLUE, ORANGE, "#1baf7a", "#52514e"
    INNER = "#86b6ef"
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(6.6, 0)
    ax.axis("off")

    def bar(x, y, w, color, text, tc="white", h=0.62, size=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, color=color, lw=0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc)

    def label(y, title, sub):
        ax.text(0.15, y + 0.2, title, fontsize=9.5, weight="bold", va="center")
        ax.text(0.15, y + 0.47, sub, fontsize=8.5, color=INK2, va="center")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2,
                                                            shrinkA=0, shrinkB=0))

    ax.text(0.15, 0.25, "Evaluation protocol: nested cross-validation, grouped by prompt", fontsize=13, weight="bold",
            va="center")
    ax.text(0.15, 0.58, "No prompt is ever on both sides of a split, because Kaggle's test prompts never appear in "
            "train.csv (a passage's nearest neighbour usually answers the same prompt).", fontsize=9, color=INK2,
            va="center")
    x0, W, gap = 3.3, 8.4, 0.04
    # row 1: train / test split
    y = 1.05
    label(y, "1 · Split once by prompt", "2,699 passages, 150 prompts")
    bar(x0, y, W * 0.8 - gap, TRAIN, "Train split · 2,159 passages · 120 prompts")
    bar(x0 + W * 0.8, y, W * 0.2, TEST, "Test split\n540 · 30 prompts", size=8.5)
    ax.text(x0 + W * 0.8 + 0.12, y + 0.85, "used once, at the end;\nalso re-scored as a Kaggle-like\nfile "
            "without prompt ids", ha="left", va="top", fontsize=8, color=INK2)
    arrow(x0 + W * 0.4, y + 0.66, x0 + W * 0.4, y + 1.0)
    # row 2: outer folds
    y = 2.15
    label(y, "2 · Outer CV, 50 folds", "5 folds × 10 repeats")
    seg = (W * 0.8) / 5
    for k in range(5):
        last = k == 4
        bar(x0 + k * seg, y, seg - gap, VAL if last else TRAIN,
            "outer val\n≈432 · 24 pr." if last else f"fold {k + 1}", size=8 if last else 9)
    ax.text(x0 + 2 * seg, y + 0.85, "outer training fold ≈1,727 passages · 96 prompts", ha="center", va="top",
            fontsize=8.5, color=INK2)
    ax.plot([x0, x0 + 4 * seg - gap], [y + 0.72, y + 0.72], color=MUTED, lw=1)
    arrow(x0 + 2 * seg, y + 1.15, x0 + 2 * seg, y + 1.45)
    # row 3: inner split
    y = 3.65
    label(y, "3 · Inner split", "80/20 of the training fold")
    wi = 4 * seg - gap
    bar(x0, y, wi * 0.8 - gap, INNER, "inner train ≈1,380", tc=INK)
    bar(x0 + wi * 0.8, y, wi * 0.2, TUNE, "inner val\n≈345", tc=INK, size=8.5)
    ax.text(x0 + wi + 0.25, y + 0.31, "every setting is trained on inner train;\nthe best on inner val is kept "
            "(C, width, dropout,\nlearning rate, epoch …)", fontsize=8.5, color=INK2, va="center")
    # row 4: steps
    y = 4.75
    steps = [("Tune", "pick the setting with the best\nbalanced accuracy on inner val", TUNE, INK),
             ("Refit", "retrain that setting on the\nwhole outer training fold", TRAIN, "white"),
             ("Score", "once on the outer validation\nfold (never used to choose)", VAL, "white"),
             ("Compare", "50 paired scores per system →\nBayesian correlated t-test,\nρ ≈ 0.2, ROPE = 1 pp", INK2,
              "white")]
    bw, bg = 2.6, 0.33
    for i, (head, body, colr, tc) in enumerate(steps):
        x = 0.15 + i * (bw + bg)
        ax.add_patch(plt.Rectangle((x, y), bw, 0.36, color=colr, lw=0))
        ax.text(x + 0.12, y + 0.18, f"{i + 1}. {head}", fontsize=9.5, weight="bold", color=tc, va="center")
        ax.add_patch(plt.Rectangle((x, y + 0.36), bw, 0.95, fc="white", ec=BASE, lw=0.8))
        ax.text(x + 0.12, y + 0.45, body, fontsize=8.5, color=INK, va="top")
        if i < len(steps) - 1:
            arrow(x + bw + 0.03, y + 0.83, x + bw + bg - 0.03, y + 0.83)
    ax.text(0.15, 6.35, "Metric: balanced accuracy (Kaggle's metric; the hidden test set is not class-balanced). "
            "Also reported: accuracy, macro-F1,\nfamily accuracy, per-class F1, accuracy by length, run time. "
            "Slow models (transformer, fine-tuning) use the first repeat only (5 folds).",
            fontsize=8.5, color=INK2, va="center")
    fig.savefig(OUT / "7_protocol.png", dpi=220, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", (OUT / "7_protocol.png").relative_to(ROOT))


# --------------------------------------------------------------------------- 8 · progression to the best result
def figure_progression():
    t7, tc, kl, clus, ft = _test_numbers()
    steps = [
        ("Length only (baseline)", "length/logreg", None),
        ("Best token features (bag-of-words)", "bow-lusr/logreg_unscaled", None),
        ("LUSR style embedding", "lusr/logreg", 100 * t7["lusr/logreg"]),
        ("LUSR fine-tuned (top 4 layers + refit)", "lusr-finetune-top4-refit/finetune", 100 * ft["refit"]["balanced_accuracy"]),
        ("Prompt-centred LUSR", "lusr-promptcentred/logreg", clus("lusr")),
        ("Prompt-centred Qwen3-Base, layer 14", "qwen3-base-L14-mean-promptcentred/logreg", clus("qwen3-base-L14-mean")),
    ]
    n = len(steps)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (label, name, test) in enumerate(steps):
        v = 100 * ba(name).mean()
        best = i == n - 1
        ax.barh(i, v, height=0.62, color="#184f95" if best else "#86b6ef", lw=0, zorder=2)
        ax.text(52.6, i, f"{v:.1f}", ha="right", va="center", fontsize=10, weight="bold" if best else "normal",
                color=INK)
        if test is not None:
            ax.scatter([test], [i], s=60, marker="D", color=ORANGE, ec="white", lw=1.2, zorder=4)
            ax.text(56.4, i, f"{test:.1f}", ha="right", va="center", fontsize=10, color=INK, weight="bold" if best else "normal")
        ax.text(-1, i, label, ha="right", va="center", fontsize=10, weight="bold" if best else "normal", color=INK)
    ax.axvline(100 / 18, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(100 / 18 + 0.5, -0.75, "chance 5.6%", fontsize=8.5, color=INK2, va="center")
    ax.text(52.6, -0.75, "CV", ha="right", va="center", fontsize=8.5, color=INK2, weight="bold")
    ax.text(56.4, -0.75, "test", ha="right", va="center", fontsize=8.5, color=INK2, weight="bold")
    ax.set_xlim(0, 56.5)
    ax.set_ylim(n - 0.4, -1.1)
    ax.set_yticks([])
    ax.set_xticks(range(0, 51, 10))
    ax.set_xlabel("balanced accuracy (%)")
    ax.grid(axis="x", color=RULE, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#86b6ef"),
               plt.Line2D([], [], marker="D", ls="", color=ORANGE, markeredgecolor="white", markersize=7)]
    ax.legend(handles, ["cross-validation (50 folds; fine-tuning 5)",
                        "held-out test split (centred: prompts recovered by clustering)"],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Best result so far: 48.7% CV, 47.6% on unseen prompts", x=0.02, ha="left", fontsize=13,
                 weight="bold", y=0.98)
    fig.text(0.02, 0.905, "Logistic regression unless stated. Removing the topic (subtracting each prompt's mean "
             "embedding) moved the score more than any model change.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.36, right=0.98, top=0.78, bottom=0.13)
    fig.savefig(OUT / "8_best_result.png", dpi=220)
    plt.close(fig)
    print("wrote", (OUT / "8_best_result.png").relative_to(ROOT))


if __name__ == "__main__":
    table_hyperparameters()
    table_models()
    table_models(SLIDE_REPS, "2b_model_comparison_slide", slide=True)
    table_cv_vs_test()
    figure_protocol()
    figure_progression()
    table_transformers()
    figure_bayes()
    figure_confusion()
