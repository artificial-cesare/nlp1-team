"""Shared data loading for NLP1 Project 1 (Name That LLM).

    from shared.data import load_data
    D = load_data()          # D.train / D.test DataFrames, D.y, D.y_test, D.groups
    X = my_features(D.train["text"])   # rows aligned with D.train -> pass to eval_harness.run_cv

Train/test = Kazik's split (GroupShuffleSplit by prompt_id, 80/20, seed 42). If the split CSVs are
missing they are rebuilt with exactly the same code, so everyone gets identical rows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "name-that-llm"
SPLIT_SEED = 42

FAMILIES = ["GPT", "Claude", "Gemini", "GLM", "Qwen", "Kimi"]
# Categorical palette (colour-blind checked), one colour per family; use for every per-family plot.
FAMILY_COLOR = dict(zip(FAMILIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))


def family(model: str) -> str:
    prefix = re.match(r"[a-z]+", model).group()
    return {"gpt": "GPT", "claude": "Claude", "gemini": "Gemini", "glm": "GLM", "qwen": "Qwen", "kimi": "Kimi"}[prefix]


@dataclass
class Data:
    full: pd.DataFrame      # all 2,699 labelled passages (train.csv)
    train: pd.DataFrame     # Kazik's train split: the only data used for CV
    test: pd.DataFrame      # Kazik's held-out split: touched once, at the end
    classes: list           # sorted class names; label id = position
    class_order: list       # classes ordered by family, then name (tables, confusion matrices)

    @property
    def class_id(self) -> dict:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def y(self) -> np.ndarray:
        return self.train["model"].map(self.class_id).to_numpy()

    @property
    def y_test(self) -> np.ndarray:
        return self.test["model"].map(self.class_id).to_numpy()

    @property
    def groups(self) -> np.ndarray:
        return self.train["prompt_id"].to_numpy()


def make_split(full: pd.DataFrame):
    """Kazik's split, verbatim from project.ipynb on main."""
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SPLIT_SEED)
    tr, te = next(gss.split(full, y=full["model"], groups=full["prompt_id"]))
    return full.iloc[tr].reset_index(drop=True), full.iloc[te].reset_index(drop=True)


def load_data(data_dir: Path = DATA_DIR) -> Data:
    full = pd.read_csv(data_dir / "train.csv")
    tr_path, te_path = data_dir / "split_train.csv", data_dir / "split_test.csv"
    if tr_path.exists() and te_path.exists():
        train, test = pd.read_csv(tr_path), pd.read_csv(te_path)
    else:
        train, test = make_split(full)
        train.to_csv(tr_path, index=False)
        test.to_csv(te_path, index=False)
    assert set(train["prompt_id"]).isdisjoint(test["prompt_id"])
    assert len(train) + len(test) == len(full)
    classes = sorted(full["model"].unique())
    class_order = sorted(classes, key=lambda m: (FAMILIES.index(family(m)), m))
    return Data(full, train, test, classes, class_order)


def length_features(texts: pd.Series) -> np.ndarray:
    """Characters, words, sentences and their logs: the length-only baseline representation."""
    texts = pd.Series(texts)
    L = np.c_[texts.str.len(), texts.str.split().str.len(), texts.str.count(r"[.!?](?:\s|$)")]
    return np.c_[L, np.log1p(L)].astype(float)
