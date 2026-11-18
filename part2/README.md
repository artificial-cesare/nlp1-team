# Part 2: word embeddings (steps 1–2)

Notebook: [`cesare_word_embeddings.ipynb`](cesare_word_embeddings.ipynb). Full written answers: [`project2_answers.md`](project2_answers.md).

## What was done

- **Compared 4 cased English embeddings:** word2vec GoogleNews, fastText Wikipedia, fastText Common Crawl, GloVe 840B.
- **Wrote one tokeniser per model** (`EmbTokenizer` in `wordvecs.py`). Each model's settings were read off its vocabulary: how it splits `don't`, whether it keeps punctuation, curly quotes, digits.
- **Measured coverage:** fastText and GloVe cover over 99.9% of our tokens. word2vec misses 22%, mostly punctuation and *a/and/of/to*, which is exactly the style signal.
- **Explored the space:** it captures meaning well and style only partly. An averaged text vector is mostly topic (the prompt explains about 65% of its variance).
- **Probed downstream:** logistic regression on mean-pooled vectors, shared 50 folds, balanced accuracy (chance 5.6%):

| Embedding | Raw | Prompt-centred |
|---|---|---|
| **fastText Common Crawl** (`ft-crawl`) | **28.5%** | **40.6%** |
| fastText Wikipedia (`ft-wiki`) | 29.6% | 39.9% |
| **GloVe 840B** (`glove-840b`) | 27.0% | 38.7% |
| word2vec (`w2v-gnews`) | 24.7% | 34.6% |

## Which embedder to use (Kazik, Liam)

1. **Main: `ft-crawl`** (fastText Common Crawl, 300d, cased). This is `WV.CHOSEN`. Use it for every experiment.
2. **Second, if you want one: `glove-840b`.** It's a different algorithm and the runner-up, so it's the more informative comparison. Skip `ft-wiki` (practically tied with `ft-crawl`, so it tells us nothing new) and `w2v-gnews` (no punctuation).

**Always load through `wordvecs.load_subset`.** It gives you the right tokeniser for that embedding, so we all tokenise the same way. Don't use the tutorial's tokeniser or GloVe loader: they lowercase everything and split contractions differently.

## Setup (once, about 5 minutes)

The embeddings live in `cache/`, which is not in git, so each of you builds them locally. From the repository root:

```bash
uv pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

python -m part2.prepare_embeddings                 # ft-crawl only (1.5 GB download)
python -m part2.prepare_embeddings glove-840b      # optional second embedder (~2 GB)
```

The script checks the files against `shared/embeddings_manifest.json` and stops if anything differs. You should see:

```
[ft-crawl] OK: 19,517 types x 300d -> cache/wordvecs/subset-ft-crawl.npz  (fingerprint c81b13dd28ef)
```

## Using it in steps 3–4

```python
import sys; sys.path.insert(0, "..")                 # if running from part2/
from part2 import embcv, wordvecs as WV
from shared import eval_harness as H, analysis as A

tok, emb = WV.load_subset(WV.CHOSEN)                 # "ft-crawl"; or WV.load_subset("glove-840b")

res = embcv.run_cv_torch(tok, emb, "embed-and-pool",  # or "lstm"
                         name="ft-crawl-pool/mlp-50x50")
H.save_result(res)                                   # -> shared/results/
A.comparison_table(H.load_results(),
                   pairs=[("emb-ft-crawl-mean/logreg", "ft-crawl-pool/mlp-50x50")])
```

- **Evaluate with `embcv.run_cv_torch`, not `part2.experiment.cross_validation`.** Ours uses the shared prompt-grouped folds, early-stops on an inner split and reports balanced accuracy, so results compare directly with the baselines in `shared/results/`. The course function picks the best epoch on the test fold and leaks prompts across folds.
- **Name results `<embedder>-<encoder>/<head>`** (e.g. `glove-840b-lstm/mlp-50x50`) so the comparison tables stay readable.
- **Useful variants:** `emb.pca(50)` for embedding size; `emb.random_like(seed=0)` for random init, plus `enc_kwargs={"freeze_embeddings": False}` to make it trainable.
- **Default is 5 folds** (1 CV repeat); pass `repeats=` for more.
- **Baselines to beat:** `emb-ft-crawl-mean/logreg` (28.5%) and `emb-ft-crawl-mean-promptcentred/logreg` (40.6%).

## Files

| File | What it is |
|---|---|
| `cesare_word_embeddings.ipynb` | Steps 1–2: the analysis behind everything above |
| `project2_answers.md` | Written answers to steps 1–2, open questions for 3–6, prioritised next steps |
| `wordvecs.py` | Loading embeddings (`load_subset`), per-model tokeniser, `pca`, `random_like` |
| `embcv.py` | `run_cv_torch`: CV for the neural models on the shared folds |
| `prepare_embeddings.py` | Downloads, builds and checks the embedding subsets |
| `../shared/embeddings_manifest.json` | Expected hashes and probe scores for each embedder |
