"""Reusable EDA helpers for the Name That LLM dataset.

Typical use in a notebook::

    from part1.eda import add_length_features, analyse_feature

    eda_df = add_length_features(train_split_df)
    eda_df["family"] = eda_df["model"].map(model_family_map)

    res = analyse_feature(eda_df, "n_words", "Words per response",
                          save_dir=EDA_GRAPHS_DIR, prefix="resp_len",
                          log_scale_model=True)
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare

_STAT_LABELS = {
    "count": "Number of texts",
    "frequency": "Texts per unit",
    "percent": "% of group's texts",
    "probability": "Proportion of group's texts",
    "proportion": "Proportion of group's texts",
    "density": "Density",
}


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _letters_per_word(s) -> float:
    """Mean token length after stripping punctuation from each whitespace token."""
    if not isinstance(s, str):
        return np.nan
    tokens = [re.sub(r"[^\w]", "", t) for t in s.split()]
    tokens = [t for t in tokens if t]
    return float(np.mean([len(t) for t in tokens])) if tokens else np.nan


def add_length_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Return a copy of ``df`` with basic length features added.

    - ``n_chars``: all characters, including spaces and punctuation
    - ``n_words``: whitespace-separated tokens
    - ``chars_per_word``: ``n_chars / n_words`` (includes spaces and punctuation)
    - ``letters_per_word``: mean token length with punctuation stripped
    """
    out = df.copy()
    text = out[text_col]
    out["n_chars"] = text.str.len()
    out["n_words"] = text.str.split().str.len()
    out["chars_per_word"] = out["n_chars"] / out["n_words"].replace(0, np.nan)
    out["letters_per_word"] = text.apply(_letters_per_word)
    return out


def summarise_by_group(
    df: pd.DataFrame,
    cols: list[str],
    by: str | list[str] = ("family", "model"),
    aggs: tuple[str, ...] = ("mean", "median", "std"),
    decimals: int = 2,
) -> pd.DataFrame:
    """Aggregate ``cols`` per group, e.g. per family and model."""
    by = list(by) if not isinstance(by, str) else by
    return df.groupby(by)[cols].agg(list(aggs)).round(decimals)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def make_palette(values, cmap: str | None = None) -> dict:
    """Fixed colour per category, so the same family has the same colour everywhere."""
    levels = sorted(pd.Series(values).dropna().unique())
    if cmap is None:
        cmap = "tab10" if len(levels) <= 10 else "husl"
    colours = sns.color_palette(cmap, n_colors=len(levels))
    return dict(zip(levels, colours))


def _finish(fig, save_path, show: bool) -> None:
    sns.despine(fig=fig)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()


def _place_legend(ax, n_levels: int, outside_after: int = 8) -> None:
    if ax.get_legend() is not None and n_levels > outside_after:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_distribution(
    df: pd.DataFrame,
    col: str,
    hue: str = "family",
    *,
    stat: str = "count",
    bins: int = 40,
    log_scale: bool = False,
    kde: bool = True,
    palette: dict | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (10, 6),
    show: bool = True,
    ax=None,
):
    """Overlaid histogram (step outline) plus KDE curve per ``hue`` group.

    ``stat="count"`` gives number of texts per bin; ``stat="percent"`` normalises
    each group separately, so groups of different sizes are comparable.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    palette = palette or make_palette(df[hue])

    sns.histplot(
        data=df,
        x=col,
        hue=hue,
        hue_order=list(palette),
        palette=palette,
        stat=stat,
        common_norm=False,
        element="step",
        fill=False,
        bins=bins,
        log_scale=log_scale,
        alpha=0.25,
        kde=kde,
        line_kws={"linewidth": 2.5},
        linewidth=2,
        ax=ax,
    )
    ax.set_xlabel(xlabel or col)
    ax.set_ylabel(_STAT_LABELS.get(stat, stat))
    ax.set_title(title or f"{xlabel or col} by {hue}")
    _place_legend(ax, len(palette))
    _finish(fig, save_path, show)
    return ax


def within_prompt_rank(df: pd.DataFrame, col: str, prompt_col: str = "prompt_id") -> pd.Series:
    """Rank of ``col`` among responses to the same prompt, scaled to [0, 1].

    0 = lowest, 1 = highest, ties get their average rank. Scaled as
    ``(rank - 1) / (n - 1)`` so a random ordering averages exactly 0.5
    (``rank(pct=True)`` would average (n + 1) / 2n, i.e. ~0.53 for 18 models).
    """
    grouped = df.groupby(prompt_col)[col]
    r = grouped.rank()
    n = grouped.transform("count")
    return (r - 1) / (n - 1)


def plot_within_prompt_rank(
    df: pd.DataFrame,
    col: str,
    *,
    model_col: str = "model",
    family_col: str = "family",
    prompt_col: str = "prompt_id",
    kind: str = "box",
    palette: dict | None = None,
    label: str | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (9, 7),
    show: bool = True,
    ax=None,
):
    """Per-model distribution of within-prompt ranks of ``col``.

    ``kind="box"`` shows the full spread; ``kind="point"`` shows mean rank with a
    95% CI (easier to read for small differences). Models are ordered by the
    median (box) or mean (point) rank. 0.5 = typical position.
    """
    if kind not in {"box", "point"}:
        raise ValueError("kind must be 'box' or 'point'")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    palette = palette or make_palette(df[family_col])

    d = df.assign(_rank=within_prompt_rank(df, col, prompt_col))
    centre = "median" if kind == "box" else "mean"
    order = d.groupby(model_col)["_rank"].agg(centre).sort_values().index

    common = dict(
        data=d,
        y=model_col,
        x="_rank",
        hue=family_col,
        hue_order=list(palette),
        palette=palette,
        order=order,
        dodge=False,
        ax=ax,
    )
    if kind == "box":
        sns.boxplot(**common)
    else:
        sns.pointplot(**common, errorbar=("ci", 95), linestyle="none")

    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"{label or col}: rank within prompt (0 = lowest, 1 = highest)")
    ax.set_ylabel(model_col)
    if title:
        ax.set_title(title)
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    _finish(fig, save_path, show)
    return ax


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def friedman_test(
    df: pd.DataFrame,
    col: str,
    *,
    model_col: str = "model",
    prompt_col: str = "prompt_id",
) -> dict:
    """Friedman test (prompts = blocks, models = treatments) plus Kendall's W.

    Prompts missing a response from any model are dropped and reported in
    ``dropped``, as ``{prompt_id: [missing models]}``.
    """
    wide = df.pivot(index=prompt_col, columns=model_col, values=col)
    incomplete = wide[wide.isna().any(axis=1)]
    dropped = {pid: row.index[row.isna()].tolist() for pid, row in incomplete.iterrows()}
    res = _friedman_from_wide(wide.dropna())
    return {
        "feature": col,
        "chi2": res["chi2"],
        "df": res["df"],
        "p": res["p"],
        "W": res["W"],
        "n_prompts": res["n"],
        "n_models": res["k"],
        "dropped": dropped,
    }


def _friedman_from_wide(wide: pd.DataFrame) -> dict:
    """Friedman test on a complete blocks x treatments table, plus Kendall's W."""
    stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
    n, k = wide.shape
    return {
        "chi2": float(stat),
        "df": k - 1,
        "p": float(p),
        "W": float(stat / (n * (k - 1))),
        "n": n,
        "k": k,
    }


def family_friedman_test(
    df: pd.DataFrame,
    col: str,
    *,
    model_col: str = "model",
    family_col: str = "family",
    prompt_col: str = "prompt_id",
) -> dict:
    """Do families differ? Friedman test on prompts x families.

    Each family's value per prompt is the mean within-prompt rank of its models.
    A significant result can be driven by a single unusual model; use
    :func:`family_permutation_test` to ask whether family membership itself
    matters.
    """
    d = df.assign(_rank=within_prompt_rank(df, col, prompt_col))
    # Only prompts answered by every model, as in the model-level test
    complete = d.groupby(prompt_col)[model_col].transform("nunique") == d[model_col].nunique()
    wide = d[complete].pivot_table(
        index=prompt_col, columns=family_col, values="_rank", aggfunc="mean"
    )
    res = _friedman_from_wide(wide.dropna())
    return {
        "feature": col,
        "chi2": res["chi2"],
        "df": res["df"],
        "p": res["p"],
        "W": res["W"],
        "n_prompts": res["n"],
        "n_families": res["k"],
    }


def family_permutation_test(
    df: pd.DataFrame,
    col: str,
    *,
    model_col: str = "model",
    family_col: str = "family",
    prompt_col: str = "prompt_id",
    n_perm: int = 10_000,
    seed: int = 0,
) -> dict:
    """Are models in the same family more alike than randomly grouped models?

    Statistic: eta squared, the share of variance in the models' mean
    within-prompt ranks that lies between families. The null distribution comes
    from shuffling family labels across models (family sizes kept fixed).
    ``eta2_chance`` is the mean of that null: with few models it is far from 0
    (about (n_families - 1) / (n_models - 1)), so compare ``eta2`` against it.
    """
    d = df.assign(_rank=within_prompt_rank(df, col, prompt_col))
    model_means = d.groupby(model_col)["_rank"].mean()
    fams = d.groupby(model_col)[family_col].first().loc[model_means.index]
    x = model_means.to_numpy()
    codes = pd.factorize(fams)[0]

    def eta2(labels: np.ndarray) -> float:
        grand = x.mean()
        sizes = np.bincount(labels)
        means = np.bincount(labels, weights=x) / sizes
        ss_between = float((sizes * (means - grand) ** 2).sum())
        return ss_between / float(((x - grand) ** 2).sum())

    observed = eta2(codes)
    rng = np.random.default_rng(seed)
    null = np.array([eta2(rng.permutation(codes)) for _ in range(n_perm)])
    return {
        "feature": col,
        "eta2": observed,
        "eta2_chance": float(null.mean()),
        "p": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
        "n_models": len(x),
        "n_families": int(codes.max() + 1),
        "n_perm": n_perm,
    }


def feature_tests(
    df: pd.DataFrame,
    cols: list[str] | str,
    *,
    model_col: str = "model",
    family_col: str = "family",
    prompt_col: str = "prompt_id",
    n_perm: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run all three tests for each feature and return one row per feature.

    Columns are grouped under ``model`` (Friedman across models), ``family``
    (Friedman across families) and ``family_perm`` (permutation test), and
    sorted by the model-level Kendall's W. The ``*_sig`` columns use ``alpha``.
    """
    cols = [cols] if isinstance(cols, str) else list(cols)
    kw = dict(model_col=model_col, prompt_col=prompt_col)
    rows = {}
    for col in cols:
        m = friedman_test(df, col, **kw)
        f = family_friedman_test(df, col, family_col=family_col, **kw)
        perm = family_permutation_test(
            df, col, family_col=family_col, n_perm=n_perm, seed=seed, **kw
        )
        rows[col] = {
            ("model", "chi2"): m["chi2"],
            ("model", "df"): m["df"],
            ("model", "p"): m["p"],
            ("model", "W"): m["W"],
            ("model", "sig"): m["p"] < alpha,
            ("family", "chi2"): f["chi2"],
            ("family", "df"): f["df"],
            ("family", "p"): f["p"],
            ("family", "W"): f["W"],
            ("family", "sig"): f["p"] < alpha,
            ("family_perm", "eta2"): perm["eta2"],
            ("family_perm", "eta2_chance"): perm["eta2_chance"],
            ("family_perm", "p"): perm["p"],
            ("family_perm", "sig"): perm["p"] < alpha,
            ("info", "n_prompts"): m["n_prompts"],
            ("info", "dropped"): m["dropped"],
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "feature"
    return out.sort_values(("model", "W"), ascending=False)


def style_feature_tests(table: pd.DataFrame):
    """Readable formatting of :func:`feature_tests` output for a notebook."""
    fmt = {}
    for c in table.columns:
        name = c[1]
        if name == "p":
            fmt[c] = "{:.2g}"
        elif name in {"chi2"}:
            fmt[c] = "{:.1f}"
        elif name in {"W", "eta2", "eta2_chance"}:
            fmt[c] = "{:.3f}"
    sig_cols = [c for c in table.columns if c[1] == "sig"]
    return (
        table.drop(columns=[("info", "dropped")])
        .style.format(fmt)
        .map(
            lambda v: "font-weight: bold; color: #1a7f37" if v is True else "",
            subset=sig_cols,
        )
        .background_gradient(subset=[("model", "W")], cmap="Greens", vmin=0, vmax=1)
    )


def format_friedman(res: dict) -> str:
    """One-line, report-style summary of :func:`friedman_test` output."""
    line = (
        f"{res['feature']}: Friedman chi2({res['df']}) = {res['chi2']:.1f}, "
        f"p = {res['p']:.3g}, Kendall's W = {res['W']:.3f} "
        f"(n = {res['n_prompts']} prompts)"
    )
    if res["dropped"]:
        line += f" | dropped incomplete prompts: {res['dropped']}"
    return line


# --------------------------------------------------------------------------- #
# All-in-one
# --------------------------------------------------------------------------- #
def analyse_feature(
    df: pd.DataFrame,
    col: str,
    label: str | None = None,
    *,
    save_dir: str | Path | None = None,
    prefix: str | None = None,
    model_col: str = "model",
    family_col: str = "family",
    prompt_col: str = "prompt_id",
    bins: int = 40,
    log_scale_family: bool = False,
    log_scale_model: bool = False,
    rank_kind: str = "box",
    show: bool = True,
) -> dict:
    """Run the standard EDA for one numeric feature.

    1. Distribution by family (percent, so family size doesn't matter)
    2. Distribution by model (count)
    3. Within-prompt rank by model
    4. Friedman test + Kendall's W (printed and returned)

    Plots are saved as ``{prefix}_by_family.png``, ``{prefix}_by_model.png`` and
    ``{prefix}_rank_by_model.png`` in ``save_dir`` when it is given.
    """
    label = label or col
    prefix = prefix or col
    save_dir = Path(save_dir) if save_dir is not None else None

    def path(name: str):
        return save_dir / f"{prefix}_{name}.png" if save_dir is not None else None

    family_palette = make_palette(df[family_col])

    plot_distribution(
        df,
        col,
        hue=family_col,
        stat="percent",
        bins=bins,
        log_scale=log_scale_family,
        palette=family_palette,
        xlabel=label,
        title=f"{label} by model family",
        save_path=path("by_family"),
        show=show,
    )
    plot_distribution(
        df,
        col,
        hue=model_col,
        stat="count",
        bins=bins,
        log_scale=log_scale_model,
        xlabel=label,
        title=f"{label} by model",
        save_path=path("by_model"),
        show=show,
    )
    plot_within_prompt_rank(
        df,
        col,
        model_col=model_col,
        family_col=family_col,
        prompt_col=prompt_col,
        kind=rank_kind,
        palette=family_palette,
        label=label,
        save_path=path("rank_by_model"),
        show=show,
    )

    res = friedman_test(df, col, model_col=model_col, prompt_col=prompt_col)
    print(format_friedman(res))
    return res
