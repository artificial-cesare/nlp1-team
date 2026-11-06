# Question 1: Dataset
resp_len_per_prompt_by_model.png
quote_style_heatmap.png
similarity_word_family.png
chars_per_word_per_prompt_by_model.png

- **Dataset**:
	- 2,699 responses.
	- 18 models (6 families).
	- 150 prompts.
	- All statistical tests ignore the one prompt for which `claude-fable-5` response is missing.
- **Split**:
	- 80/20 by prompt.
	- All EDA and tokeniser evaluation use the training split only; the held-out test set is reserved for the final model evaluation.
- **Statistics**:
	- Response length carries no signal
	- Punctuation identifies the family
		- Em-dash rate is the strongest feature; family membership explains 77-97% of between-model variance for em-dashes, colons, hyphens and semicolons.
	- Few signals separate models within a family
		- word length
		- commas
		- full stops
		- second-person pronouns
	- Same-family models write more similar answers to the same prompt:
		- TF-IDF cosine 0.286
		- permutation p < 0.001 on all 119 prompts
- **Conclusions**:
	- Surface features should identify the family well; separating models within a family needs richer representations (word or character n-grams, stylistic features, punctuation, etc.).
	- Kept 14 features: em-dash, letters per word, colon, of/an ration, hedging, comma, semicolon, hyphen, full stop, contractions, number words, second person, digits, parentheses, quote style etc.
	- Preprocessing: Keep Unicode punctuation and case; normalising quotes or dashes would erase the strongest cues.

# Question 2: Tokenisation
whole_word_by_frequency.png
fertility_vs_vocab.png

- Our BPE:
	- keeps every punctuation cue as its own token, unlike the pretrained tokenisers
	- needs more tokens per word and splits rare words
	- We hand over our BPE at 2k and 4k (chosen) and 8k (comparison closest to the pretrained tokenisers) for the downstream accuracy to evaluate.
	- Punctuation isolated in pre-tokenisation, no normaliser (case kept), full character alphabet.
- `roberta-bpe` - model trained for style and authorship representations
- `qwen3-bpe` - large vocabulary of a recent LLM family
- NFC Text preprocessing is identical for all tokenisers. It changes none of the training texts.
- Cross validation for evaluation, metrics used:
	- UNK rate (UNK tokens / tokens, and responses with any UNK)
	- fertility (tokens per whitespace word)
	- whole-word rate (words covered by exactly one token)
	- cue preservation (share of each cue character that survives encode + decode and is a distinct token)

|Tokeniser|Vocab|Responses with UNK|Fertility|Train–val gap|Whole-word|Cue distinct|
|---|---|---|---|---|---|---|
|ours 2k|2,000|1.1%|2.01|2.7%|0.55|1.00|
|ours 4k|4,000|1.1%|1.72|5.6%|0.67|1.00|
|ours 8k|8,000|1.1%|1.50|9.0%|0.80|1.00|
|`roberta-bpe`|50,265|0|1.29|–|0.91|0.84|
|`qwen3-bpe`|151,669|0|1.28|–|0.91|0.53|

- **Conclusions**:
	- UNK is a sanity check (rare characters missing from training folds) - byte-level tokenisers never produce UNK.
	- All tokenisers kept frequent words whole, but ours keeps only 1-2% of words unseen in the training fold whole vs. about 45% for the pretrained tokenisers.
