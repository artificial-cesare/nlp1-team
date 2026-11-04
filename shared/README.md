# shared/: one protocol for every representation

All three tracks (Kazik: EDA and tokenisers; Liam: token indicators and bag-of-words; Cesare: neural embeddings) import from here, so every number is computed on the same data, folds, models, grids, metric and statistical test. **Do not edit `folds.json`.** Change models or grids only after agreeing as a team, then re-run.

| File | What it gives you |
|---|---|
| `data.py` | `load_data()`: Kazik's train/test split (rebuilt identically if the CSVs are missing), labels `D.y`, prompt groups `D.groups`, `D.classes`, `D.class_order` (by family), `family()`, `FAMILY_COLOR`, `length_features()` |
| `tokenization.py` | `preprocess()` (NFC only), `tokenize(texts, name)`, `tokenizer_stats()`, the `TOKENIZERS` registry |
| `eval_harness.py` | `load_folds()`, `MODELS` (model specs and hyperparameter grids), `run_cv()` (nested CV, parallel over folds), `compare()` (Bayesian correlated t-test, ROPE = 1 pp), `final_test()`, `save_result()` / `load_results()` |
| `analysis.py` | `results_table`, `comparison_table`, `plot_posteriors`, `plot_confusion` (family-ordered), `family_errors`, `per_class_f1`, `accuracy_by_length`, `misclassified` |
| `text_models.py` | Neural-track models that read text directly: `run_cv_text(..., "transformer" or "finetune")` (tiny transformer from scratch; fine-tuned LUSR) and `final_test_text()`; same result format as `run_cv` |
| `folds.json` | Stratified 5-fold CV grouped by `prompt_id`, 10 repeats (50 folds), by passage `id` |
| `results/` | One JSON per system (`<representation>__<model>.json`): per-fold scores, chosen params, out-of-fold predictions. The neural track's results are already here (best: `lusr/logreg`, 35.8% balanced accuracy) |

## Protocol (details and justification: `part1/neural_project.ipynb`, step 2)

- **Data:** CV uses Kazik's train split only (2,159 passages, 120 prompts). The test split (540 passages, 30 prompts) is used once, at the end, via `final_test()`.
- **Folds:** grouped by prompt, because Kaggle's test prompts are unseen. 10 repeats for logistic regression and the MLPs; costly neural models use the first repeat only, and `compare()` then pairs them on those 5 folds.
- **Tuning:** nested. Inside each outer training fold, 20% of its prompts form an inner validation set; the best grid point (and, for MLPs, the early-stopping epoch) is refit on the whole outer training fold.
- **Metric:** balanced accuracy (the Kaggle metric) is primary; accuracy and macro-F1 are also reported.
- **Comparison:** Bayesian correlated t-test (Benavoli et al. 2017), ρ = n_test / n_total ≈ 0.2, ROPE = 1 percentage point. It matches `baycomp.two_on_single`.

## Models and grids (`eval_harness.MODELS`)

| Name | Architecture | Grid |
|---|---|---|
| `logreg` | StandardScaler + multinomial logistic regression (log-linear) | C ∈ {1e-4, 1e-3, 1e-2, 0.1, 1, 10} |
| `logreg_ovr` | One-vs-rest logistic regression (cheap check only) | C = 0.01 |
| `mlp` | Standardise → Dropout → Linear → ReLU → Dropout → Linear | hidden {256, 1024} × dropout {0.2, 0.5} × lr {1e-3, 3e-4} × weight decay {1e-4, 1e-2} |
| `resmlp` | Projection + 2–3 pre-LayerNorm residual blocks + LayerNorm + linear head | layers {2, 3} × hidden {256, 512} × dropout {0.3, 0.5} × lr {1e-3, 3e-4}, weight decay 1e-2 |

MLPs use AdamW, batch 64, early stopping with patience 15 on inner balanced accuracy (at most 200 epochs), seed 42. Sparse inputs are densified for the MLPs; for a very wide vocabulary, cap it (for example `max_features`) inside your vectoriser. Fit vectorisers inside each fold if they learn from the data (for example IDF or a vocabulary), or say why not.

## Minimal example

```python
import sys; sys.path.insert(0, "..")          # from part1/, so `shared` is importable
from shared.data import load_data
from shared import eval_harness as H, analysis as A

D = load_data()
FOLDS = H.load_folds(D.train["id"].tolist())   # 50 (repeat, fold, train_idx, test_idx)
X = my_features(D.train["text"])               # rows aligned with D.train; dense or scipy sparse
for model in ["logreg", "mlp", "resmlp"]:
    H.save_result(H.run_cv(X, D.y, D.groups, FOLDS, model, name=f"bow/{model}"))

R = H.load_results()
A.results_table(R)
table, cmps = A.comparison_table(R, baseline="bow/logreg", pairs=[("bow/mlp", "bow/resmlp")])
```

Timing on an M-series Mac (10 cores, 1,024-dimensional input, 50 folds): logreg about 7 s, mlp about 2.6 min, resmlp about 4.3 min, all on CPU (folds in parallel; MPS is slower for MLPs this small). Time your representation on one fold first (`FOLDS[:1]`).
