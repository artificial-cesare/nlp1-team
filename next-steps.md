# Next steps

A list of ideas for next week. Add new ideas at the bottom.

## Where we stand (neural track, 2026-11-04)

- Best CV balanced accuracy is about 36% on 18 classes (chance is 5.6%). The best system is frozen LUSR embeddings with logistic regression.
- About 70% of predictions get the model family right. About half of the errors pick the wrong model inside the right family.
- MLP and ResMLP on the same embeddings score within 1 pp of logistic regression. The representation limits us, not the classifier.
- Fine-tuning LUSR reaches 33.8% (5 epochs) and 33.5% (10 epochs). Validation loss is lowest at epoch 3, then rises while training loss falls. With about 80 passages per class, the model memorises the training set.
- Results are in `part1/neural_project.ipynb` and `shared/results/`.

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
