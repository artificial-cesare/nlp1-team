"""Prompt-centring and prompt recovery (notebook step 7b).

Every prompt is answered by many models, so most of the variation between passage embeddings is topic.
`prompt_centre` subtracts each prompt group's mean embedding, leaving how each author differs from the others on
the same topic. It uses group ids only, never labels.

Kaggle's test.csv has no prompt_id, so at test time the groups must be recovered: `cluster_prompts` does
average-linkage agglomerative clustering on cosine distances of a topic embedding (Qwen3-Embedding works best),
with a distance threshold tuned on the training split (`tune_threshold`), so the number of prompts is not needed.
`kaggle_like_subsample` mimics the hidden test set, where the number of passages per model is not equal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score


def prompt_centre(X, groups, min_size: int = 1):
    """X minus the mean of the rows in the same group. Groups smaller than `min_size` are centred with the mean
    of all rows instead (a group of one would otherwise become the zero vector)."""
    X = np.asarray(X, dtype=np.float64)
    groups = pd.Series(np.asarray(groups))
    means = pd.DataFrame(X).groupby(groups.values).transform("mean").to_numpy().copy()
    small = groups.map(groups.value_counts()).to_numpy() < min_size
    means[small] = X.mean(0)
    return (X - means).astype(np.float32)


def _linkage(Z):
    return linkage(pdist(np.asarray(Z, dtype=np.float64), metric="cosine"), method="average")


def cluster_prompts(Z, threshold: float):
    """Cluster ids from average-linkage clustering of Z (cosine distance), cut at `threshold`."""
    if len(Z) < 2:
        return np.zeros(len(Z), dtype=int)
    return fcluster(_linkage(Z), threshold, criterion="distance")


def tune_threshold(Z, prompt_ids, grid=np.arange(0.20, 0.99, 0.005)) -> tuple[float, float]:
    """Distance threshold that maximises the adjusted Rand index against the true prompt ids. Returns (thr, ARI)."""
    L = _linkage(Z)
    scores = [(adjusted_rand_score(prompt_ids, fcluster(L, t, criterion="distance")), t) for t in grid]
    ari, thr = max(scores)
    return float(thr), float(ari)


def kaggle_like_subsample(y, rng, keep=(0.25, 1.0)):
    """Indices of a subsample with unequal passages per class: class k keeps each passage with probability
    p_k ~ Uniform(keep). Prompt groups become incomplete, as in the hidden test set."""
    y = np.asarray(y)
    p = rng.uniform(*keep, size=y.max() + 1)
    return np.flatnonzero(rng.random(len(y)) < p[y])
