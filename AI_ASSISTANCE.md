# AI assistance log

We disclose AI assistance as the course requires. Tool: Claude Code (Claude Opus 5.5), run locally on the `cecce-pizza` branch. One line per step: what the AI did, and what we decided.

- **Step 0 (setup):** AI found branch `cecce-pizza`. It found that Kazik's split exists only as one cell in main's `project.ipynb` (GroupShuffleSplit by `prompt_id`, 80/20, seed 42), and it added `transformers`, `sentence-transformers` and `accelerate`. Cesare decided to copy the split cells word for word into the notebook (commit `1f82fa9`) instead of checking out main's notebook, which would have overwritten his step-1 work.
