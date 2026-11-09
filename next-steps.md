# Next steps

A list of ideas for next week. Add new ideas at the bottom of the matching section.

## Where we stand (neural track, 2026-11-05)

- **Best result so far: 48.7% CV balanced accuracy** on 18 classes (chance is 5.6%), and 47.6% on the held-out test split with prompts recovered by clustering, as Kaggle requires (46.9% on Kaggle-like subsamples with unequal classes). It comes from Qwen3-0.6B-Base layer-14 embeddings with each prompt's mean subtracted, then logistic regression (notebook step 7b). Centred LUSR reaches 44.1% CV (46.9% test with recovered clusters). Without the centring, LUSR + logistic regression reaches 35.8% and Qwen3-Base 34.2%.
- MLP and ResMLP on the same embeddings score within 1 pp of logistic regression. The representation limits us, not the classifier.
- Full fine-tuning of LUSR reaches 33.8% (5 epochs) and 33.5% (10 epochs). It overfits after epoch 3. Training only the top 4 of 12 layers reaches 35.8% (cell 6.5c), level with logistic regression. Refitting that model on the whole training fold for the chosen number of epochs raises it to 37.5% CV and 35.7% test (cell 6.5d): more data helps fine-tuning.
- We have about 120 passages per class in the train split (about 96 in each inner training set).
- Results are in `part1/neural_project.ipynb` and `shared/results/`.

## Findings from the per-class analysis

- **What the inputs look like.** Each text is one paragraph of creative prose, about 110 words, with no markdown. In RoBERTa tokens the median is 130, the 95th percentile 285 and the maximum 385. Each of the 150 prompts is answered once by each of the 18 models. Most of the variation between texts comes from the prompt (topic), not from the author.
- **Hard classes are the ones with a near-twin.** Per-class F1 has the same ranking in every system. It correlates with:
  - how pure each class's 10 nearest neighbours in LUSR space are (Spearman 0.75);
  - the cosine to the nearest other class centroid (-0.56);
  - the share of errors that go to one partner class (-0.65).
- **Claude Fable 5, GLM-5.3 and GLM-5.3-flash are almost identical.** The Fable and GLM-5.3 centroids have cosine 0.976. All three use unspaced em dashes and straight apostrophes heavily, and none has distinctive words. GLM writes like Claude, which explains the GLM-to-Claude errors.
- **Easy classes have a fingerprint.**
  - Opus 4.6 and Opus 5 are the only models that write a spaced em dash (" — ", in 26 to 29% of their texts). They also write long sentences with few commas.
  - Gemini 3.1 Pro has favourite words ("profound", "absolute", "microscopic") and never uses em dashes.
  - The GPT family uses curly apostrophes (’); Claude and GLM never do.
- **Hand-made typography counts** (17 counts of dashes, quotes, punctuation and sentence length) reach 26% balanced accuracy alone. Added to LUSR they give +0.6 pp, inside the ROPE, so LUSR already captures most of them.

## 1. Checks before changing anything

1. **Family accuracy and within-family accuracy for each method.** This shows whether the problem is the representation or a ceiling. *Done (out-of-fold, first CV repeat):*

   | Method | Balanced accuracy | Family accuracy | Within-family accuracy |
   |---|---|---|---|
   | LUSR, prompt-centred + logistic regression | 44.1% | 75.2% | 58.7% |
   | LUSR + logistic regression | 35.8% | 69.5% | 52.8% |
   | LUSR fine-tuned, top 4 layers | 35.8% | 71.6% | 50.1% |
   | LUSR fine-tuned, all layers, 10 epochs | 33.5% | 72.9% | 46.0% |
   | Qwen3 L14 + logistic regression | 34.2% | 64.1% | 53.3% |

   Fine-tuning gets the **family** right more often than logistic regression, but the **model within the family** less often. Its extra capacity goes into family differences, not the within-family distinctions where the errors are. The family-aware ideas (B4) target exactly this.
2. **Did both pipelines see the same tokens?** *Done:* yes. No passage is longer than 385 tokens, so neither the frozen embeddings nor fine-tuning truncates anything at 512. The comparison is fair.
3. **Learning curve.** Train logistic regression on LUSR and the top-4 fine-tuning on 25%, 50%, 75% and 100% of each training fold. If fine-tuning's curve is steeper, more data is the lever (D1). The plot is a good backup slide. *Not done yet.*
4. **Liam's character n-gram or TF-IDF logistic regression on the same folds.** If word distribution carries most of the signal, a well-tuned bag-of-words model may match LUSR, and the two may make different errors. Check the overlap of their errors before stacking (C1). *Waiting for Liam's results in `shared/results/`.*

## 2. Ideas

### A. Use the structure of the data (tested on 2026-11-04, LUSR + logistic regression, 50 folds)

1. **Remove the topic: subtract each prompt's mean embedding.** Balanced accuracy goes from 35.8% to 44.1% (+8.3 pp, P(better) = 1.00). The result is saved as `shared/results/lusr-promptcentred__logreg.json`.
2. **One-to-one assignment within a prompt.** Each prompt has exactly one answer per model, so assign the 18 answers to the 18 models with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on the log-probabilities). With centring this reaches 48.2% (first CV repeat), and family accuracy goes from 69.5% to 76.5%. Fable's F1 goes from 0.16 to 0.26.
3. **For Kaggle,** the test file has no prompt ids, and the number of passages per model is not equal (so groups are incomplete). Clustering Qwen3-Embedding vectors (threshold tuned on train) recovers the prompts with ARI 0.988 on our test split; `shared/prompt_groups.py` has the helpers and notebook step 7b.3 the simulation.
4. **For the TA,** present A1 and A2 as a use of the dataset's design, not as a better model.
5. **Next:** A1 is now notebook step 7b, for LUSR and both Qwen3 embeddings (Qwen3-Base gains most: 34.2% → 48.7%). Still to do: add A2 to the notebook, try the MLPs on centred embeddings, and apply the centring to the fine-tuned model's pooled vectors.

### B. Making the fine-tuned model pay off

1. **Train on chunks, predict per document.** The label applies to every sentence of a passage. Train on random crops each epoch and average the logits over all windows at inference. This multiplies the training examples and attacks the variance problem directly. Keep the folds at passage level, so that chunks of one passage never sit on both sides of a split.
   - Our passages are short (median 130 tokens), so 128 to 256-token windows give only one or two chunks. Use windows of about 64 tokens, or two to three sentences, to get 2 to 4 chunks per passage.
2. **LP-FT (linear probing, then fine-tuning).** Initialise the head from the fitted logistic regression on frozen LUSR. Use a backbone learning rate about 10 times smaller than the head's, or layer-wise learning-rate decay. Then unfreeze the layers one by one, from the top. This starts fine-tuning from our best model instead of from a random head.
3. **LoRA instead of layer freezing.** LoRA moves the backbone less than unfreezing whole layers, which is the direction our top-2 and top-4 results point (`pip install peft`). Also try 8 to 10 epochs: with partial freezing, the best epoch was usually the last one.
4. **Use the family structure.**
   - Add an auxiliary family head (6 classes) to the 18-way loss.
   - Or train with a supervised contrastive loss in which family members count as weak positives (as in DeTeCtive), then fit logistic regression or kNN on the resulting space.
   - Either way, the model's capacity goes into the within-family distinctions where the errors are (check 1). A simpler variant is a two-stage classifier: predict the family first, then the model inside that family.
5. **Prompt-aware fine-tuning.**
   - Build each batch from all the answers to a few prompts. Subtract the prompt mean of the pooled vectors inside the network (A1 inside the model).
   - Train with cross-entropy or a contrastive loss: same model on a different prompt = positive; same prompt, different model = negative.
   - This teaches the encoder author differences instead of topic.
6. **Attention pooling instead of mean pooling,** so that a single rare cue (one spaced em dash in 130 tokens) is not averaged away.

### C. Fuse and add other signals

1. **Fuse instead of replacing.** Stack out-of-fold probabilities under a logistic regression meta-learner. Candidates:
   - LUSR + logistic regression (prompt-centred);
   - the fine-tuned model;
   - Qwen3;
   - Liam's bag-of-words and character n-gram models.

   Three methods at about 36% that make different errors can beat each of them. This is cheap, because the CV results already exist.
2. **Likelihood features from small local LMs.** Score each passage with a few small open models: Qwen3-0.6B (already loaded), plus small Llama, Gemma and Mistral models. Use the mean and variance of surprisal, entropy and rank as features. A model tends to find text from its own family less surprising. Estimated cost: about 30 min on MPS.
3. **Better embeddings without training.** Average several Qwen3 layers instead of only layer 14, or extract embeddings with a larger model (Qwen3-1.7B or 4B), which is feasible on MPS.

### D. More data

1. **More passages per class.** Most of the levers above are a substitute for this. ATLAS with 1,000 texts per source already beat a discriminative baseline trained on 30k; we have about 100. If the learning curve (check 3) is still rising at 100%, generating more passages is the best use of time. Use the same 18 models, the same kind of prompt, one answer per model per prompt, and check each provider's terms of service.

## Housekeeping

- **Tokenisers.** Only `roberta-bpe` and `qwen3-bpe` feed our models (`USED_TOKENIZERS` in `shared/tokenization.py`). `modernbert`, `deberta-v3` and `tutorial-bpe-1k` appear only in the comparison table. Drop them if the rubric does not ask for a comparison.
- **Kazik** to confirm RoBERTa BPE as the tokeniser for the from-scratch transformer.
- **Liam** to send Kazik his tokeniser for the indicator and bag-of-words features, so Kazik can evaluate it too.
- **Merging `main`.** `.gitignore` conflicts. Keep both blocks: our `cache/` and Kazik's `graphs/` plus his setup files.
- **Kaggle leaderboard.** Not checked yet. Compare our CV score with the public leaderboard.
- **Cosmetic.** The step 6.5b summary includes an averaged "fold" row that means nothing. Remove it.
- More Kaggle ideas: the last cell of `part1/neural_project.ipynb`.
