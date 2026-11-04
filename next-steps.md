# Next steps

A list of ideas for next week. Add new ideas at the bottom.

## Where we stand (neural track, 2026-11-04)

- Best CV balanced accuracy is about 36% on 18 classes (chance is 5.6%). The best system is frozen LUSR embeddings with logistic regression.
- About 70% of predictions get the model family right. About half of the errors pick the wrong model inside the right family.
- MLP and ResMLP on the same embeddings score within 1 pp of logistic regression. The representation limits us, not the classifier.
- Fine-tuning LUSR reaches 33.8% (5 epochs) and 33.5% (10 epochs). Validation loss is lowest at epoch 3, then rises while training loss falls. With about 80 passages per class, the model memorises the training set.
- Results are in `part1/neural_project.ipynb` and `shared/results/`.

## Findings from the per-class analysis (2026-11-04)

- **What the inputs look like.** Each text is one paragraph of creative prose, about 110 words (about 150 tokens, so nothing is truncated), with no markdown. Each of the 150 prompts is answered once by each of the 18 models. Most of the variation between texts comes from the prompt (topic), not from the author.
- **Hard classes are the ones with a near-twin.** Per-class F1 has the same ranking in every system. It correlates with how pure each class's 10 nearest neighbours in LUSR space are (Spearman 0.75), and negatively with the cosine to the nearest other class centroid (-0.56) and the share of errors that go to one partner class (-0.65).
- **Claude Fable 5, GLM-5.3 and GLM-5.3-flash are almost identical.** The Fable and GLM-5.3 centroids have cosine 0.976. All three use unspaced em dashes and straight apostrophes heavily, and none has distinctive words. GLM writes like Claude, which explains the GLM-to-Claude errors.
- **Easy classes have a fingerprint.** Opus 4.6 and Opus 5 are the only models that write a spaced em dash (" — ", in 26 to 29% of their texts), and they write long sentences with few commas. Gemini 3.1 Pro has favourite words ("profound", "absolute", "microscopic") and never uses em dashes. The GPT family uses curly apostrophes (’); Claude and GLM never do.
- **Hand-made typography counts** (17 counts of dashes, quotes, punctuation and sentence length) reach 26% balanced accuracy alone. Added to LUSR they give +0.6 pp, inside the ROPE, so LUSR already captures most of them.

## New top ideas (tested on 2026-11-04, LUSR + logistic regression, 50 folds)

0. **Remove the topic: subtract each prompt's mean embedding.** Balanced accuracy goes from 35.8% to 44.1% (+8.3 pp, P(better) = 1.00). The result is saved as `shared/results/lusr-promptcentred__logreg.json`.
0b. **One-to-one assignment within a prompt.** Each prompt has exactly one answer per model, so assign the 18 answers to the 18 models with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment` on the log-probabilities). With centring this reaches 48.2% (repeat 0), and family accuracy goes from 69.5% to 76.5%.
0c. **For Kaggle,** the test file has no prompt ids. TF-IDF clustering recovers the prompts: 99.1% of texts land in their prompt's cluster (ARI 0.91). Check that the hidden test set also contains complete groups of 18 answers per prompt.
0d. **Prompt-aware fine-tuning.** Build each batch from all the answers to a few prompts, subtract the prompt mean of the pooled vectors inside the network, and train with cross-entropy or a contrastive loss (same model, different prompt = positive; same prompt, different model = negative). This teaches the encoder author differences instead of topic. Replace mean pooling with attention pooling, so that a single rare cue (one spaced em dash) is not averaged away.
- Chunk augmentation (idea 3) matters less than expected: the texts are short single paragraphs.

## Ideas, by expected gain for our data size

1. **Combine the feature sets (stacking).** Stack the out-of-fold probabilities of LUSR, Qwen3 and Liam's character n-gram and stylometric features under a logistic regression meta-model. Style cues (quotes, markdown, punctuation) separate models in the same family, and embeddings smooth them out. Cheap, because the CV results already exist.
2. **Likelihood features from small local LMs.** Score each passage with a few small open models: Qwen3-0.6B (already loaded), plus small Llama, Gemma and Mistral models. Use the mean and variance of surprisal, entropy and rank as features. A model tends to find text from its own family less surprising. Estimated cost is about 30 min on MPS.
3. **Chunk augmentation.** Split each passage into 2 to 4 chunks for training, and average the predictions per passage. This gives fine-tuning more examples.
4. **Fine-tuning with fewer trainable parameters.** *Tried on 2026-11-04 (cell 6.5c): training only the top 4 of 12 layers gives 35.8%, up from 33.8% with full fine-tuning and level with logistic regression. Next: 8 to 10 epochs, LoRA (`pip install peft`), and chunk augmentation (idea 3).* Use LoRA or freeze most layers. Alternatively, use contrastive training followed by logistic regression or kNN. These should overfit less than full fine-tuning. This is the most likely way for fine-tuning to beat logistic regression, but it needs more compute.
5. **Better embeddings without training.** Average several Qwen3 layers instead of only layer 14. Alternatively, extract embeddings with a larger model (Qwen3-1.7B or 4B), which is feasible on MPS.
6. **Two-stage classifier.** Predict the family first, then the model inside that family.

## Housekeeping

- **Tokenisers.** Only `roberta-bpe` and `qwen3-bpe` feed our models (`USED_TOKENIZERS` in `shared/tokenization.py`). `modernbert`, `deberta-v3` and `tutorial-bpe-1k` appear only in the comparison table. Drop them if the rubric does not ask for a comparison.
- **Kazik** to confirm RoBERTa BPE as the tokeniser for the from-scratch transformer.
- **Liam** to send Kazik his tokeniser for the indicator and bag-of-words features, so Kazik can evaluate it too.
- **Merging `main`.** `.gitignore` conflicts. Keep both blocks: our `cache/` and Kazik's `graphs/` plus his setup files.
- **Kaggle leaderboard.** Not checked yet. Compare our CV score with the public leaderboard.
- **Cosmetic.** The step 6.5b summary includes an averaged "fold" row that means nothing. Remove it.
- More Kaggle ideas: the last cell of `part1/neural_project.ipynb`.
