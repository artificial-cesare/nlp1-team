# Project 2: answers (steps 1–2) and open questions (steps 3–6)

Evidence for every claim below: `part2/cesare_word_embeddings.ipynb`. Numbers are balanced accuracy on the shared 50 folds (chance 5.6%) unless stated. "P" is the posterior probability from the team's Bayesian correlated t-test, with ROPE = 1 pp.

**Start here (Liam, Kazik):** section [Reproduce](#reproduce-before-steps-36) at the end gets you the same embeddings and numbers in about 5 minutes.

---

## 1. Find a word embedding model that is appropriate for your dataset

> Think about the language, the kind of text you are working with. Find evidence that your choice is a good one.

**Answer: fastText trained on Common Crawl (`crawl-300d-2M`, 300d, cased).**

Our texts are English creative prose in which **who wrote it** shows up in punctuation, quotes, dashes, case and function words, not in the topic. So we need a model that:
- is **cased**;
- **keeps punctuation as tokens**;
- has seen **web-like text**.

We compared four candidates that are cased and English:
- word2vec GoogleNews;
- fastText Wikipedia+news;
- fastText Common Crawl;
- GloVe 840B Common Crawl.

fastText had the best performance (Common Crawl and Wikipedia are practically tied; we take Common Crawl for slightly better coverage and the best prompt-centred score), GloVe 840B a close second, and word2vec was clearly the worst.

### Can you tokenize your texts in a way coherent with the definition of a token in the embedding model?

**Yes, but not with the tutorial's tokeniser, and the rules differ per model.** What the documentation says:
- **fastText** (both English releases): the paper names the tokeniser, the Moses MT project's `tokenizer.perl` (Mikolov et al. 2018, *Advances in Pre-Training Distributed Word Representations*). The download page says nothing.
- **GloVe:** the paper says the corpora were tokenised *and lowercased* with the Stanford tokeniser (Pennington et al. 2014). The 840B release is cased, so that description does not fully apply to it; the release itself is documented only as "cased".
- **word2vec GoogleNews:** no preprocessing is documented at all.

The papers name tools but not their settings, so we checked every rule against the vocabulary (a word's row number is its frequency rank, so "absent" means the training tokeniser never produced it). Docs and vocabulary agree. For example, Moses splits `don't` into `don` + `'t`, keeps `well-known` whole, and treats a curly `’` as ordinary punctuation, so `’` becomes a token of its own. That is why fastText has `’` but no `’s` or `’t`.

| | word2vec | fastText | GloVe 840B |
|---|---|---|---|
| Case | kept | kept | kept |
| Punctuation | **none at all** | every mark is a token (`—` `’` `“` `;` …) | every mark is a token, but typographic ones are rare |
| `don't` | `don't` (whole) | `don` + `'t` | `do` + `n't` |
| `it’s` (curly) | — | absent; we fold it to `'s` or keep `’` as its own token | `it` + `’s` (has curly forms) |
| `a`, `and`, `of`, `to` | **removed** (`the` mostly removed too: rank 11, below `The`) | kept | kept |
| Digits | `1990` → `####` | kept | kept |

We wrote one tokeniser (`part2/wordvecs.py`, `EmbTokenizer`) that never changes the text, only how a token is looked up. Each model gets its own settings, which are stored in `WV.TOKENIZER_CONFIG`.

The tutorial pipeline lowercases everything, turns `"` into ` `` `/`''`, and produces `n't`, which fastText doesn't have.

### Does the model's vocabulary give enough coverage to the words in your texts?

**fastText and GloVe: yes. word2vec: no.**
- **fastText Common Crawl:** 99.95% of tokens have a vector. Only 6% of texts contain any unknown word, and those are rare science words such as *chemolithotrophic* and *Göbekli*.
- **GloVe:** 99.92% of tokens, 9% of texts.
- **word2vec has no vector for 22% of our tokens:** 13.5% are punctuation and 8% are *a/and/of/to*. That is exactly the style signal, and the loss even differs by author (20–24%).
- **Every cue from our initial EDA survives in fastText:** all 1,476 em dashes, every `;` `:` `,` and quote. The exception is the curly apostrophe, which survives only if we don't fold `’s` into `'s`.

---

## 2. Explore what your chosen word embedding model captures

### Is it good at representing interesting similarities (re to your choice of task)?

**Partly: it represents *meaning* well and *style* only partly.**
- **Meaning works:** *king − man + woman ≈ queen*; the neighbours of *essay* are *essays, dissertation, thesis*.
- **The cues are separate points in the space.**
  - In fastText, `’` vs `'` has cosine 0.36 and `—` vs `-` 0.22. That is above two random words (0.10) but far from synonyms (*big/large* 0.69), so the model can tell curly from straight and em dash from hyphen.
  - In GloVe, curly and straight quotes are almost synonyms (0.74), so averaging blurs the cue.
- **Neighbourhoods reflect the job a mark does:** `—` sits near `;`, *But*, *And* (it joins clauses), and `’` sits near `'s`, `'ll`, `'ve` (it marks contractions).
- **Some authors have a recognisable vocabulary region.** Gemini's favourite words (*profound, absolute, abyss, tapestry, primordial*) cluster together, as do Opus 4.6/4.8/5's and GLM-5.3's. GPT, Fable and GLM-flash's do not, and Fable and GLM-flash are the hardest classes from part 1.
- **But an averaged text vector is mostly topic:** the prompt explains about 65% of its variance, the author only 3–5%. Averaging only function words and punctuation shifts the balance towards the author.

### Have you found task-specific evidence that it's better than some other reasonable alternative?

**Yes.** We tested a logistic regression on the averaged vectors, on the shared folds:

| Embedding | Raw | Prompt-centred |
|---|---|---|
| **fastText Common Crawl** | **28.5%** | **40.6%** |
| fastText Wikipedia | 29.6% | 39.9% |
| GloVe 840B | 27.0% | 38.7% |
| word2vec | 24.7% | 34.6% |

- **fastText vs word2vec:** +3 to +5.5 pp, P = 0.92–1.00.
- **fastText vs GloVe:** +1.6 to +1.9 pp, P = 0.66–0.79.
- **The two fastText corpora are practically equivalent.**
- **Against part 1:** averaged fastText beats our BPE bag-of-words (21.7%, +6.8 pp, P ≈ 1.00) and character n-grams (26.0%, +2.5 pp, P = 0.88). It is still below LUSR (35.8%; 44.1% prompt-centred).
- **Curly quotes:** folding them is the only choice that hurts (P(fold better) = 0.004), so we keep `’` as its own token (`curly="mark"`).

### Do you know about limitations of this model?

**Yes; they apply to all static embeddings (fastText, GloVe, word2vec):**
- **One vector per word, whatever the context:** *like* the verb and *like* the preposition share a vector, and so does an em dash in any role.
- **Averaging loses word order:** *dog bites man* = *man bites dog*.
- **Meaning, not style:** the models were trained to predict neighbouring words, so similar means "used in similar contexts", which mostly means similar topic.
- **Old corpora (2013–2017), before LLMs existed:** nothing in the space says that LLMs overuse *delve* or *tapestry*.
- **Unknown words:**
  - the `.vec` files have no subword vectors, so rare words get nothing (about 6% of texts);
  - GloVe and word2vec never had subwords.
- **Model-specific:**
  - word2vec dropped punctuation and the commonest function words;
  - GloVe 840B has few typographic characters in training, conflates curly and straight quotes, and its crawl has encoding noise;
  - all of them inherit social biases (Bolukbasi et al. 2016).

### Any limitations that you find particularly troubling for your task?

**Three:**
1. **Topic drowns style.** The model groups texts by what they are about, and in our data that is the prompt, which says nothing about the author. Subtracting each prompt's mean vector fixes much of it (+12 pp).
2. **Averaging dilutes rare cues.** One em dash is about 1% of a text's tokens, so about 1% of its vector.
3. **What the tokeniser can't see is lost.** Whitespace is not a token, so Opus's **spaced** em dash (" — ") looks the same as an unspaced one.

---

## 3. Experiment with the embed-and-pool approach. — **@LIAM**
* What design choices have meaningful impact in classification performance (eg, embedding size, architecture design, optimisation parameters), if any?
* Is this better than what you could achieve last week? In other words, did pretrained embeddings add to your design?

_Answer (Liam; notebook `part2/liam_embed_and_pool.ipynb`). Frozen fastText Common Crawl, balanced accuracy, shared folds, Bayesian correlated t-test with ROPE 1 pp. Cesare's first draft is in git history (01cd4f3)._

**First bullet: which design choices matter.** What goes into the pooled vector matters; the classifier on top does not.

| Choice | Effect | Folds | Decided? |
|---|---|---|---|
| Prompt-centring the pooled vector (same MLP) | **+13.4 pp** (27.4% → 40.8%) | 5 | yes, P = 0.999 |
| Function-word + punctuation channel next to the mean (logreg) | **+2.4 pp** (40.6% → 43.0%) | 50 | nearly, P = 0.94 |
| Embedding size: PCA 50 / 100 / 200 vs 300 (logreg) | **−6.1 / −5.6 / −4.5 pp** | 50 | yes, P ≥ 0.999: all 300 dims are needed |
| Adding max pooling to mean + func (logreg) | −1.4 pp | 50 | doesn't help (P(better) = 0.01) |
| Epochs 20 → 100, tutorial MLP (`embcv`, not centred) | 19.8% → 23.1%; best epoch now 24–65 | 5 | was undertrained, now converged |
| MLP vs logreg, centred mean | −0.4 pp | 50 | likely equal, P(ROPE) = 0.68 |
| MLP (nested-tuned) vs logreg, centred mean + func | +0.6 pp (43.6% vs 43.0%) | 50 | likely equal, P(ROPE) = 0.65 |
| Tuned MLP / ResMLP vs logreg, centred mean | +0.7 / −1.4 pp | 5 | undecided |

- Upward embedding sizes can't be tested: the cased releases are 300d only.
- The tutorial MLP pipeline (`embcv`) trails the harness MLP on the same raw vectors (23.1% vs 27.4%, 5 folds, not tested): the harness standardises inputs and refits on the full training fold.

**Best embed-and-pool system:** mean of all tokens + mean of function words/punctuation, prompt-centred, logistic regression: **43.0%** (50 folds).

**Second bullet: did pretrained embeddings help vs part 1?** Yes against part 1's bag-of-words features, no against its contextual embeddings (same 50 folds):

| Part 1 system | Part 1 | Ours (43.0%) − part 1 | Decided? |
|---|---|---|---|
| BPE bag-of-words (`bow-ours4k`) | 21.7% | +21.3 pp | yes, P ≈ 1.00 |
| Character 2–4-grams | 26.0% | +17.0 pp | yes, P ≈ 1.00 |
| LUSR | 35.8% | +7.2 pp | yes, P ≈ 1.00 |
| LUSR, prompt-centred | 44.1% | −1.0 pp | undecided (P(ROPE) = 0.42) |
| Qwen3-Base L14, prompt-centred | 48.7% | −5.7 pp | yes, part 1 better (P = 0.999) |

Part of the gain over bag-of-words and raw LUSR comes from centring, which those baselines lack. Like for like, raw fastText (28.5%) beats BPE bag-of-words (+6.8 pp, P ≈ 1.00), probably beats character n-grams (+2.5 pp, P = 0.88), and trails raw LUSR by 7.3 pp (P ≈ 1.00).

## 4. Experiment with a recurrent encoder (GRU or LSTM). — **@KAZIK**
* What design choices have meaningful impact in classification performance, if any?
* Did you investigate the impact of data size, embedding parameters (trained or not, fixed or trainable, etc)?
* Did you struggle with over-fitting?
* How far can this approach go? Are you able to improve meaningfully on embed-and-pool?

_Answer:_

## 5. We make decisions under uncertainty all the time, are you gathering enough information to substantiate your choices? — **@LIAM & @KAZIK**
Various things can lead to small differences in results, and these differences are not always consistent. Here are some things to think about:
* How did you structure you hyperparameter search?
* Are you keeping an eye on experimental variance?

_Answer:_

## 6. Experiment with error analysis and try to discover what characteristics of input texts are predictive of performance differences across major design choices. — **@LIAM & @KAZIK**
* What analysis features did you come up with?
* Which ones did you manage to test and how did that go?
* What would you prioritise in the future in order to improve your systems?

_Answer:_

---

## Cesare's plan vs the notebook: discrepancies

The plan was written before the notebook; the notebook was written from `part2/TODO.md` ("Word2Vec and fastText"). This table compares them point by point.

| Topic | Plan | Notebook | Who was right, and what changed |
|---|---|---|---|
| Algorithms | fastText + GloVe | first word2vec + fastText; **GloVe added after this comparison** | **The plan was right** that GloVe belongs: it is the runner-up (−1.6 pp vs fastText). word2vec stays as a useful negative example (it shows what losing punctuation costs). Both algorithms from the plan are now compared, plus word2vec. |
| "Versions in which punctuation is preserved" | yes | yes, plus case | **Both right**, and this turned out to be the most important criterion (word2vec −3 to −5.5 pp). The plan didn't mention case, which matters just as much: the tutorial's GloVe pipeline lowercases everything. |
| Punctuation glued to words as separate embeddings (`hence,`) | idea | tested | **No such tokens exist.** None of the tables has `hence,` or `however,`. fastText's ~21k `word.` types are noise (`light.`), not a convention we could use. Use separate punctuation tokens. |
| ≥ 2 hyperparameters per algorithm (dims, `minn`/`maxn`) | yes | corpus/size tested for fastText (wiki 1M vs crawl 2M: equivalent) | **Partly feasible.** The official cased fastText and GloVe releases exist only at 300d, and `minn`/`maxn` only matter when *training* fastText. Workable size axis: `emb.pca(k)` (same model, k dims; added for step 3), or GloVe 6B 50/100/200/300d (uncased, so case would be lost too). |
| Train our own embeddings | if quick | not done | **Not worth it as a separate model.** We have about 290k tokens; the pretrained models saw 16–840 *billion*. Vectors trained on our data alone would be poor. Step 4's "random initialisation, trainable" *is* training our own embeddings, end to end on the task, so that cell of the grid answers the question. (gensim also doesn't install on Python 3.14.) |
| Tokeniser "from the docs" | read docs | read the **vocabulary** | **Both are needed, and they agree.** The papers name the tools (fastText: Moses `tokenizer.perl`; GloVe: Stanford tokeniser) but not their settings, and word2vec documents nothing. The vocabulary confirms each rule and shows what the docs leave out: fastText splits `don't` into `don` + `'t` (not NLTK's `do` + `n't`), and word2vec deleted *a/and/of/to*. |
| Coverage "from Kazik's EDA" | use EDA stats | measured coverage directly, and checked every EDA cue | **Complementary.** Kazik's EDA gives the list of what must survive; coverage has to be measured with the embedding's own vocabulary. Status of his 14 features below. |
| "Can any of the 2 recover order? `<hence,>` vs `<, hence>`" | question | answered | **No, neither can.** A static model gives *hence* one vector wherever it is, and averaging ignores order, so both orders give the same text vector. Order needs step 4 (LSTM) or n-gram features. This is a good minimal-pair test for Kazik's LSTM. |
| Task-specific evidence "from the previous subquestion" | similarities | a downstream probe (logistic regression on the shared folds, Bayesian test) | **The probe is the stronger evidence:** similarity examples are anecdotes, the probe measures the task. |
| Limitations of fastText and GloVe | list | listed and ranked by how much they hurt this task | Same direction; GloVe's curly/straight conflation is a new finding. |
| Hugging Face "word embedders" | maybe | not tried | Open. The static models there are mostly the same GloVe/fastText files. Model2Vec models (static vectors distilled from sentence transformers) are a real alternative, but the ones we know of use uncased subword tokenisers, which would cost case. |

**Kazik's 14 EDA features: do they survive in fastText Common Crawl?**

| Survives | Feature |
|---|---|
| ✓ | em dash, colon, semicolon, comma, full stop, parentheses, hyphen (as `-` or inside words), digits, number words, second-person pronouns, hedging words, *of/an* ratio |
| ✓ only with `curly="mark"` | quote style (curly vs straight) |
| ✓ split into two tokens | contractions (`don` + `'t`) |
| ✗ | letters per word: word length is not encoded |

word2vec loses most of the punctuation features, the digits and *of*.

### Mistakes in the notebook along the way (all fixed)

- **GloVe left out at first.** That followed `TODO.md`, and your plan was right; it's now included.
- **The distinctive-words test was biased.** It gave every author the same words because of a vocabulary mismatch in the log-odds. It's fixed, and the corrected run is what's reported.
- **The small hand-off table missed a few keys.** For 3 word2vec texts, the subset tokeniser split differently from the full table. The new check caught it and it's fixed: the subset now keeps every key a lookup found.
- **The tutorial tokeniser is not harmful downstream.** The notebook first argued it was. For *averaged* vectors it ties with ours (inside the ROPE). Ours is still the right choice (it keeps case, `'t` and the curly cue), but the case for it is coherence with the model and step 4, not a measured gain in step 3.
- **An unverified claim about the tutorial's GloVe file.** The notebook stated it is lowercased; what we know for sure is that the tutorial's loader lowercases every key.

---

## What to improve next (prioritised)

1. **Evaluate steps 3–6 with `part2/embcv.py`, not `part2.experiment.cross_validation`.** The course function:
   - picks the best epoch on the same fold it reports (optimistic);
   - uses folds that are not grouped by prompt (answers to one prompt leak across folds);
   - reports accuracy, not balanced accuracy.

   `embcv.run_cv_torch` uses the shared prompt-grouped folds, early-stops on an inner split, reports balanced accuracy, and returns the shared result format, so `H.compare` against the probes works. It is seeded and gives identical results on re-runs (tested).
2. **The tutorial's default MLP head is undertrained on our data.** With 20 epochs the best epoch was the last or nearly the last in every fold (17–19). On the same 5 folds it is 9.7 pp below logistic regression on the same vectors (19.8%, P(logreg better) = 0.997). Start step 3 by giving it more epochs or a higher learning rate, and standardise the inputs, before comparing designs.
3. **Prompt-centring inside the neural models.** Subtract each prompt's mean pooled vector: +12 pp for the probe; part 1 step 7b has the details.
4. **A style channel.** Pool function words and punctuation separately (alone they already match all tokens, +5 pp over content words). Add a marker token for spaced em dashes, since whitespace is invisible otherwise.
5. **A negation strategy.** Static vectors don't encode what negation *means*:
   - `not` is just another function word (its nearest neighbours in fastText are `necessarily`, `is`, `be`);
   - opposites sit close together (`good`/`bad` 0.74);
   - averaging hides negation: "The result is good." vs "The result is not good." has cosine 0.975, about the same as "The result is good." vs "The result is bad." (0.968).

   For authorship, the habit matters more than the meaning, and the habit differs between authors (counts on our training texts):
   - **Gemini negates about half as often** as every other family: 5.9–6.9 negations per 1,000 tokens vs 11.3–15.5. Negations counted: *not, no, never, nothing, nor, none, nobody, neither, nowhere, cannot* and `n't`.
   - **Claude and GLM contract most:** `n't` is 4–16% of `n't` + `not` (Sonnet 5 highest at 16%). GPT, Gemini, Qwen and Kimi contract rarely (0–2%).

   Our tokenisers keep negation as its own token (`'t` in fastText, `n't` in GloVe) and keep `don 't` apart from `do not`. So the pooled vector holds this signal, but only weakly: one token is about 1/134 of the mean. What to try, in order:
   - **Explicit features:** negation rate and contraction share added next to the pooled vector. It's cheap; test it in the 2.5 probe first.
   - **Minimal pair for step 4:** check whether the LSTM separates "The result is good." from "The result is not good.", which averaging cannot do.
   - **Negation marking** (Das & Chen 2001; Pang et al. 2002): prefix `NOT_` to every word after a negation, up to the next punctuation mark. `NOT_` tokens have no pretrained vector, so this needs trainable embeddings, e.g. each initialised from its plain word.

   One known quirk: fastText splits `won't` into `won` + `'t`, and `won` is the vector for *win*. GloVe's `wo` + `n't` avoids this. It's rare in our texts.
6. **Pooling that doesn't dilute rare cues:** max, mean+max, or IDF-weighted mean.
7. **Embedding size:** `emb.pca(50/100/200)` against 300, with everything else fixed.
8. **Second algorithm for step 3:** GloVe 840B (`WV.load_subset("glove-840b")`) is ready.

---

## Reproduce (before steps 3–6)

From the repository root, on this branch:

```bash
uv pip install -r requirements.txt                 # part2 needs jax, numpyro, arviz, textstat (in requirements.txt)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

python -m part2.prepare_embeddings                 # ft-crawl only: 1.5 GB download, ~2 min
python -m part2.prepare_embeddings --all --probe   # all four tables (~6 GB) + re-run the step 2.5 probe
```

What you should see:

```
[ft-crawl] OK: 19,517 types x 300d -> cache/wordvecs/subset-ft-crawl.npz  (fingerprint c81b13dd28ef)
emb-ft-crawl-mean/logreg               bal.acc 0.2853 ± 0.0212  (50 folds)
emb-ft-crawl-mean-promptcentred/logreg bal.acc 0.4060 ± 0.0243  (50 folds)
```

What the script checks:
- the downloaded files' sha256 and the subset fingerprints against `shared/embeddings_manifest.json` (tracked in git);
- the probe scores against the manifest;
- that the small subset gives **exactly** the same token ids and vectors as the full table.

If anything differs, it stops and says what. We rebuilt `ft-crawl` from scratch and got the same fingerprint and scores.

**Using the embeddings in steps 3–4:**

```python
import sys; sys.path.insert(0, "..")                 # from part2/
from part2 import embcv, wordvecs as WV
from shared import eval_harness as H, analysis as A

tok, emb = WV.load_subset(WV.CHOSEN)                 # "ft-crawl"; or "glove-840b", "ft-wiki", "w2v-gnews"
res = embcv.run_cv_torch(tok, emb, "embed-and-pool", name="ft-crawl-pool/mlp-50x50",   # or "lstm"
                         enc_kwargs={"freeze_embeddings": True},
                         regressor_kwargs={"hidden_size1": 50, "hidden_size2": 50, "dropout": 0.1},
                         opt_kwargs={"learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "num_epochs": 20})
H.save_result(res)                                   # -> shared/results/

R = H.load_results()
A.comparison_table(R, pairs=[("emb-ft-crawl-mean/logreg", "ft-crawl-pool/mlp-50x50")])

emb_small  = emb.pca(50)                             # embedding size (step 3)
emb_random = emb.random_like(seed=0)                 # pretrained vs random init (step 4); with
                                                     # enc_kwargs={"freeze_embeddings": False} = trainable
```

- `run_cv_torch` uses 1 of the 10 CV repeats (5 folds) by default; pass `repeats=` for more.
- `H.compare` pairs results on their shared repeats automatically.
- Everything runs on CPU and is seeded; GPU runs may differ in the last decimals.
