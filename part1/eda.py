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
    """Friedman test on a complete blocks x treatments table, plus Kendall's W.

    Returns NaNs when no block has any variation (e.g. a mark nobody uses).
    """
    n, k = wide.shape
    if n == 0 or wide.nunique(axis=1).max() <= 1:
        nan = float("nan")
        return {"chi2": nan, "df": k - 1, "p": nan, "W": nan, "n": n, "k": k}
    stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
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
        return ss_between / ss_total

    ss_total = float(((x - x.mean()) ** 2).sum())
    if ss_total == 0 or np.isnan(ss_total):
        nan = float("nan")
        return {
            "feature": col,
            "eta2": nan,
            "eta2_chance": nan,
            "p": nan,
            "n_models": len(x),
            "n_families": int(codes.max() + 1),
            "n_perm": 0,
        }
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


# --------------------------------------------------------------------------- #
# Punctuation
# --------------------------------------------------------------------------- #
# name -> regex. Counted per 100 words by add_punctuation_features().
PUNCT_MARKS: dict[str, str] = {
    "comma": r",",
    "period": r"(?<!\.)\.(?!\.)",  # single dots only; "..." counted separately
    "semicolon": r";",
    "colon": r":",
    "em_dash": "—",  # —
    "en_dash": "–",  # –
    "hyphen": r"(?<=\w)-(?=\w)",  # word-internal: well-known, long-term
    "spaced_hyphen": r"(?<=\s)-(?=\s)",  # " - " used as a dash
    "paren": r"\(",
    "question": r"\?",
    "exclaim": r"!",
    "ellipsis_char": "…",  # …
    "ellipsis_dots": r"\.\.\.",
}

# name -> regex. Used for "does this response use X at all" shares.
QUOTE_MARKS: dict[str, str] = {
    "curly_apos": "’",  # ’ (also closing single quote)
    "straight_apos": r"'",
    "curly_dquote": "[“”]",  # “ ”
    "straight_dquote": r'"',
}


def add_punctuation_features(
    df: pd.DataFrame,
    text_col: str = "text",
    words_col: str = "n_words",
    marks: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add ``<mark>_per100`` (rate per 100 words) for each mark, raw counts of each quote character
    (``n_<quote>``), and a categorical ``quote_style``.

    ``quote_style`` is ``curly``, ``straight``, ``mixed`` or ``none`` based on
    every apostrophe and double quote in the response. Run
    :func:`add_length_features` first (``n_words`` is needed).
    """
    marks = marks or PUNCT_MARKS
    out = df.copy()
    text = out[text_col]
    words = out[words_col].replace(0, np.nan)

    for name, pat in marks.items():
        out[f"{name}_per100"] = text.str.count(pat) / words * 100

    for name, pat in QUOTE_MARKS.items():
        out[f"n_{name}"] = text.str.count(pat)

    curly = out["n_curly_apos"] + out["n_curly_dquote"]
    straight = out["n_straight_apos"] + out["n_straight_dquote"]
    out["quote_style"] = np.select(
        [(curly > 0) & (straight > 0), curly > 0, straight > 0],
        ["mixed", "curly", "straight"],
        default="none",
    )
    out.loc[text.isna(), "quote_style"] = np.nan
    return out


def punct_cols(marks: dict[str, str] | None = None) -> list[str]:
    """Column names produced by :func:`add_punctuation_features` for ``marks``."""
    return [f"{name}_per100" for name in (marks or PUNCT_MARKS)]


def usage_share(
    df: pd.DataFrame,
    count_cols: list[str],
    by: str | list[str] = ("family", "model"),
) -> pd.DataFrame:
    """Share of responses per group with a count > 0 in each column."""
    by = list(by) if not isinstance(by, str) else by
    return (df[count_cols] > 0).groupby([df[b] for b in np.atleast_1d(by)]).mean()


def quote_style_table(df: pd.DataFrame, by: str | list[str] = ("family", "model")) -> pd.DataFrame:
    """Share of each group's responses per ``quote_style`` (rows sum to 1), plus the number of
    responses with any quote character (``n_with_quotes``).

    Shares are over responses that use at least one quote character, so
    ``none`` is reported only as a count.
    """
    by = list(by) if not isinstance(by, str) else by
    has_q = df[df["quote_style"].isin(["curly", "straight", "mixed"])]
    shares = pd.crosstab(
        [has_q[b] for b in np.atleast_1d(by)],
        has_q["quote_style"],
        normalize="index",
    ).reindex(columns=["curly", "straight", "mixed"], fill_value=0.0)
    shares["n_with_quotes"] = has_q.groupby(list(np.atleast_1d(by))).size()
    return shares


def quote_style_test(df: pd.DataFrame, model_col: str = "model") -> dict:
    """Chi-square test of independence (model x quote style) plus Cramér's V.

    Uses only responses containing a quote character. Responses to the same prompt are not
    independent, so treat p as descriptive; V (0 = no association, 1 = style fully determined by
    model) is the useful number.
    """
    from scipy.stats import chi2_contingency

    has_q = df[df["quote_style"].isin(["curly", "straight", "mixed"])]
    table = pd.crosstab(has_q[model_col], has_q["quote_style"])
    table = table.loc[:, table.sum() > 0]
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if min(table.shape) > 1 else 0
    return {
        "chi2": float(chi2),
        "df": int(dof),
        "p": float(p),
        "cramers_v": float(v),
        "n_responses": int(n),
    }


def plot_model_heatmap(
    table: pd.DataFrame,
    *,
    labels: dict[str, str] | None = None,
    fmt: str = ".2f",
    scale: str = "column",
    cmap: str = "Blues",
    title: str | None = None,
    cbar_label: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = True,
    ax=None,
):
    """Heatmap of a (model x feature) table, annotated with the raw values.

    ``scale="column"`` colours each column by its own min-max, so rare and
    common marks are equally readable (colour = relative, numbers = absolute).
    ``scale="global"`` uses one colour scale for the whole table (use it for
    shares that are already on 0-1). A (family, model) MultiIndex is shown as
    "model (family)" and kept in family order.
    """
    t = table.copy()
    if isinstance(t.index, pd.MultiIndex):
        t.index = [f"{m} ({f})" for f, m in t.index]
    if labels:
        t = t.rename(columns=labels)

    if scale == "column":
        rng_ = (t.max() - t.min()).replace(0, np.nan)
        colour = ((t - t.min()) / rng_).fillna(0)
        vmin, vmax = 0, 1
        cbar_label = cbar_label or "relative within column (0 = min, 1 = max)"
    elif scale == "global":
        colour, vmin, vmax = t, None, None
    else:
        raise ValueError("scale must be 'column' or 'global'")

    if ax is None:
        w = max(6, 0.9 * t.shape[1] + 4)
        h = max(4, 0.38 * t.shape[0] + 1.5)
        fig, ax = plt.subplots(figsize=figsize or (w, h))
    else:
        fig = ax.figure
    sns.heatmap(
        colour,
        annot=t.to_numpy(),
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": cbar_label} if cbar_label else None,
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return ax


# --------------------------------------------------------------------------- #
# Distinctive vocabulary: log-odds ratio with informative Dirichlet prior
# (Monroe, Colaresi & Quinn, 2008, "Fightin' Words")
# --------------------------------------------------------------------------- #
VOCAB_TOKEN_PATTERN = r"(?u)\b[a-z]+(?:'[a-z]+)?\b"


def _normalise_for_vocab(s: str) -> str:
    """Lowercase and map curly quotes to straight ones, so "it’s" and "it's" are the same word here
    (quote style is analysed separately)."""
    return s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _log_odds_z(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Z-scores of each group (row) vs.

    all other rows, per term (column).
    """
    a0 = alpha.sum()
    total = counts.sum(axis=0)
    n = counts.sum(axis=1, keepdims=True)
    y_i, y_j = counts, total - counts
    n_i, n_j = n, n.sum() - n
    l_i = np.log((y_i + alpha) / (n_i + a0 - y_i - alpha))
    l_j = np.log((y_j + alpha) / (n_j + a0 - y_j - alpha))
    var = 1.0 / (y_i + alpha) + 1.0 / (y_j + alpha)
    return (l_i - l_j) / np.sqrt(var)


def _distinctive_block(
    texts: pd.Series,
    groups: pd.Series,
    *,
    vec_kwargs: dict,
    k: int,
    prior_scale: float,
    min_count: int,
    min_docs: int,
) -> pd.DataFrame:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer

    vec = CountVectorizer(**vec_kwargs)
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()

    levels = sorted(groups.unique())
    codes = pd.Categorical(groups, categories=levels).codes
    ind = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(levels), len(codes)),
    )
    counts = np.asarray((ind @ X).todense(), dtype=float)
    docs = np.asarray((ind @ (X > 0).astype(float)).todense())
    n_resp = np.bincount(codes, minlength=len(levels))

    keep = counts.sum(axis=0) >= min_count
    counts, docs, vocab = counts[:, keep], docs[:, keep], vocab[keep]
    if counts.shape[1] == 0:
        return pd.DataFrame()

    alpha = prior_scale * counts.sum(axis=0)
    z = _log_odds_z(counts, alpha)

    n_tok = counts.sum(axis=1)
    rest_counts = counts.sum(axis=0) - counts
    rest_tok = n_tok.sum() - n_tok

    rows = []
    for g, level in enumerate(levels):
        ok = np.where((docs[g] >= min_docs) & (z[g] > 0))[0]
        top = ok[np.argsort(-z[g, ok])][:k]
        for rank, t in enumerate(top, start=1):
            rows.append(
                {
                    "group": level,
                    "rank": rank,
                    "term": vocab[t],
                    "z": z[g, t],
                    "count": int(counts[g, t]),
                    "per10k": counts[g, t] / n_tok[g] * 1e4,
                    "per10k_rest": rest_counts[g, t] / rest_tok[g] * 1e4,
                    "n_responses": int(docs[g, t]),
                    "share_responses": docs[g, t] / n_resp[g],
                }
            )
    return pd.DataFrame(rows)


def distinctive_terms(
    df: pd.DataFrame,
    group_col: str = "model",
    text_col: str = "text",
    *,
    within: str | None = None,
    k: int = 10,
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: str | list[str] | None = None,
    prior_scale: float = 0.1,
    min_count: int = 10,
    min_docs: int = 5,
) -> pd.DataFrame:
    """Most over-used terms per group, ranked by log-odds z-score.

    Each group (e.g. a model) is compared against all other rows; with
    ``within="family"`` each model is compared only against the other models
    of its own family (single-model families are skipped). This is the
    comparison that matters once punctuation has identified the family.

    The prior is ``prior_scale`` x the corpus counts of each term (an
    informative Dirichlet prior), which shrinks rare terms towards zero.
    |z| > 1.96 is roughly significant, but with thousands of terms treat z as a
    ranking, not a test.

    Filters: a term must occur ``min_count`` times overall and in at least
    ``min_docs`` of the group's responses, so a word from a single prompt
    can't top the list. Because every model answers the same prompts, topic
    words largely cancel out and what is left is style.

    Columns: ``group, rank, term, z, count, per10k`` (rate per 10,000 tokens in
    the group), ``per10k_rest`` (in the comparison set), ``n_responses`` and
    ``share_responses`` (responses of the group that contain the term), plus
    ``family`` when ``within`` is used.
    """
    vec_kwargs = dict(
        token_pattern=VOCAB_TOKEN_PATTERN,
        lowercase=False,  # done in _normalise_for_vocab
        ngram_range=ngram_range,
        stop_words=stop_words,
    )
    params = dict(
        vec_kwargs=vec_kwargs,
        k=k,
        prior_scale=prior_scale,
        min_count=min_count,
        min_docs=min_docs,
    )
    d = df.dropna(subset=[text_col, group_col])
    texts = d[text_col].map(_normalise_for_vocab)

    if within is None:
        return _distinctive_block(texts, d[group_col], **params)

    parts = []
    for fam, idx in d.groupby(within).groups.items():
        sub = d.loc[idx]
        if sub[group_col].nunique() < 2:
            continue
        block = _distinctive_block(texts.loc[idx], sub[group_col], **params)
        if not block.empty:
            parts.append(block.assign(family=fam))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out[["family"] + [c for c in out.columns if c != "family"]]


def top_terms_summary(table: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    """One row per group with its top terms as a comma-separated string."""
    t = table if k is None else table[table["rank"] <= k]
    keys = ["family", "group"] if "family" in t.columns else ["group"]
    return (
        t.sort_values(keys + ["rank"])
        .groupby(keys, sort=True)["term"]
        .agg(", ".join)
        .rename("top terms")
        .to_frame()
    )


def plot_distinctive_terms(
    table: pd.DataFrame,
    groups: list[str] | None = None,
    *,
    k: int = 10,
    col_wrap: int = 3,
    palette: dict | None = None,
    group_to_colour: dict | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Small multiples: horizontal bars of the top-``k`` z-scores per group.

    ``group_to_colour`` maps each group to a colour key in ``palette`` (e.g.
    ``model -> family`` with a family palette), so bars share family colours.
    """
    groups = groups or sorted(table["group"].unique())
    n = len(groups)
    ncols = min(col_wrap, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 0.32 * k * nrows + 1.2 * nrows),
        squeeze=False,
    )
    for ax, g in zip(axes.flat, groups):
        sub = table[(table["group"] == g) & (table["rank"] <= k)].sort_values("z")
        colour = "#4C72B0"
        if palette and group_to_colour:
            colour = palette.get(group_to_colour.get(g), colour)
        ax.barh(sub["term"], sub["z"], color=colour)
        ax.set_title(g, fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("log-odds z", fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    _finish(fig, save_path, show)
    return axes  # not fig: Jupyter would display a returned Figure a second time


# --------------------------------------------------------------------------- #
# Model similarity: TF-IDF cosine between responses to the same prompt
# --------------------------------------------------------------------------- #
WORD_TFIDF_KWARGS: dict = dict(
    token_pattern=VOCAB_TOKEN_PATTERN,
    lowercase=False,  # done in _normalise_for_vocab
    sublinear_tf=True,
    min_df=2,
)
CHAR_TFIDF_KWARGS: dict = dict(
    analyzer="char",
    ngram_range=(2, 4),
    lowercase=False,  # keep case, punctuation and quote style
    sublinear_tf=True,
    min_df=2,
)


def within_prompt_similarity(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    model_col: str = "model",
    prompt_col: str = "prompt_id",
    analyzer: str = "word",
    idf: str = "global",
    vec_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Mean TF-IDF cosine similarity between models, compared only within prompts.

    With ``idf="global"`` one TF-IDF vectoriser (vocabulary + IDF) is fitted on
    all responses in ``df``, so every prompt is embedded in the same space (as a
    classifier would see it). With ``idf="prompt"`` a separate vectoriser is
    fitted on each prompt's responses, so words that every model uses for that
    prompt are down-weighted (robustness check). For each prompt answered by
    every model, the cosine similarity between the models' responses gives a
    k x k matrix; the result is the mean over those prompts. Prompts missing a
    response are skipped, as in :func:`friedman_test`.

    ``analyzer="word"`` uses lowercased word unigrams with curly apostrophes
    merged (as in :func:`distinctive_terms`); ``analyzer="char"`` uses raw
    character 2-4-grams, which keep punctuation, case and quote style.
    ``vec_kwargs`` overrides the defaults.

    Returns ``(mean_sim, per_prompt)``: a model x model DataFrame (diagonal = 1)
    and the stacked per-prompt matrices, shape ``(n_prompts, k, k)``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if analyzer == "word":
        kwargs = {**WORD_TFIDF_KWARGS, **(vec_kwargs or {})}
        texts = df[text_col].map(_normalise_for_vocab)
    elif analyzer == "char":
        kwargs = {**CHAR_TFIDF_KWARGS, **(vec_kwargs or {})}
        texts = df[text_col]
    else:
        raise ValueError("analyzer must be 'word' or 'char'")
    if idf not in {"global", "prompt"}:
        raise ValueError("idf must be 'global' or 'prompt'")

    X = TfidfVectorizer(**kwargs).fit_transform(texts) if idf == "global" else None
    models = sorted(df[model_col].unique())
    pos = np.arange(len(df))

    mats = []
    for _, idx in df.groupby(prompt_col).indices.items():
        sub = df.iloc[idx]
        if sub[model_col].nunique() < len(models) or len(sub) != len(models):
            continue
        order = pos[idx][np.argsort(sub[model_col].to_numpy())]  # rows in model order
        if idf == "global":
            Xp = X[order]  # rows are L2-normalised
        else:
            Xp = TfidfVectorizer(**{**kwargs, "min_df": 1}).fit_transform(texts.iloc[order])
        mats.append(cosine_similarity(Xp))
    per_prompt = np.stack(mats)
    mean_sim = pd.DataFrame(per_prompt.mean(axis=0), index=models, columns=models)
    mean_sim.index.name = mean_sim.columns.name = model_col
    return mean_sim, per_prompt


def double_centre(sim: pd.DataFrame) -> pd.DataFrame:
    """Remove each model's overall similarity level from a similarity matrix.

    Some models are similar to everyone ("hubs"), others to no one. Double
    centring (``s_ij - mean_i - mean_j + grand mean``, off-diagonal means)
    keeps only pair-specific affinity: 0 = as similar as the two models'
    general levels predict. The diagonal is set to 1 for plotting.
    """
    s = sim.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(s, np.nan)
    row = np.nanmean(s, axis=1)
    c = s - row[:, None] - row[None, :] + np.nanmean(s)
    np.fill_diagonal(c, 1.0)
    return pd.DataFrame(c, index=sim.index, columns=sim.columns)


def family_similarity_table(sim: pd.DataFrame, model_to_family: dict) -> pd.DataFrame:
    """Family x family mean of the model similarities (diagonal excludes self-pairs).

    Single-model families have no within-family pair, so their diagonal is NaN.
    """
    fams = pd.Series(model_to_family).loc[sim.index].to_numpy()
    s = sim.to_numpy().copy()
    np.fill_diagonal(s, np.nan)
    levels = sorted(set(fams))
    out = pd.DataFrame(index=levels, columns=levels, dtype=float)
    for a in levels:
        for b in levels:
            block = s[np.ix_(fams == a, fams == b)]
            out.loc[a, b] = np.nanmean(block) if np.isfinite(block).any() else np.nan
    return out


def family_similarity_permutation_test(
    sim: pd.DataFrame,
    model_to_family: dict,
    *,
    n_perm: int = 10_000,
    seed: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Are models in the same family more similar than models in different families?

    Statistic: mean similarity over same-family model pairs minus mean over
    different-family pairs (upper triangle of ``sim``, self-pairs excluded). The
    null distribution comes from shuffling family labels across models (family
    sizes fixed), which permutes whole rows and columns of the matrix together
    (a Mantel-style test), so the dependence between pairs that share a model is
    kept.

    Returns ``(overall, per_family)``. ``overall`` holds the within / between
    means, their difference, the null mean and the one-sided p-value.
    ``per_family`` has, for each family with 2+ models, its mean within-family
    similarity and the permutation p-value for that family's label (how often
    a random set of the same size is at least as similar).
    """
    models = list(sim.index)
    fams = pd.Series(model_to_family).loc[models]
    codes, levels = pd.factorize(fams)
    s = sim.to_numpy()
    iu = np.triu_indices(len(models), k=1)
    pair_sim = s[iu]

    def stats(labels: np.ndarray) -> tuple[float, np.ndarray]:
        same = labels[iu[0]] == labels[iu[1]]
        diff = pair_sim[same].mean() - pair_sim[~same].mean()
        per_fam = np.array(
            [
                pair_sim[same & (labels[iu[0]] == f)].mean() if sizes[f] > 1 else np.nan
                for f in range(len(levels))
            ]
        )
        return diff, per_fam

    sizes = np.bincount(codes)
    same_obs = codes[iu[0]] == codes[iu[1]]
    obs_diff, obs_fam = stats(codes)

    rng = np.random.default_rng(seed)
    null_diff = np.empty(n_perm)
    null_fam = np.empty((n_perm, len(levels)))
    for i in range(n_perm):
        null_diff[i], null_fam[i] = stats(rng.permutation(codes))

    overall = {
        "within_mean": float(pair_sim[same_obs].mean()),
        "between_mean": float(pair_sim[~same_obs].mean()),
        "diff": float(obs_diff),
        "diff_chance": float(null_diff.mean()),
        "diff_null_sd": float(null_diff.std()),
        "p": float((np.sum(null_diff >= obs_diff) + 1) / (n_perm + 1)),
        "n_within_pairs": int(same_obs.sum()),
        "n_between_pairs": int((~same_obs).sum()),
        "n_perm": n_perm,
    }
    rows = []
    for f, fam in enumerate(levels):
        if sizes[f] < 2:
            continue
        rows.append(
            {
                "family": fam,
                "n_models": int(sizes[f]),
                "within_mean": obs_fam[f],
                "chance_mean": float(np.nanmean(null_fam[:, f])),
                "p": float((np.sum(null_fam[:, f] >= obs_fam[f]) + 1) / (n_perm + 1)),
            }
        )
    return overall, pd.DataFrame(rows).set_index("family")


def nearest_neighbours(sim: pd.DataFrame, model_to_family: dict) -> pd.DataFrame:
    """Most similar other model for each model, and whether it is in the same family."""
    a = sim.to_numpy(copy=True)
    np.fill_diagonal(a, -np.inf)
    s = pd.DataFrame(a, index=sim.index, columns=sim.columns)
    nn = s.idxmax(axis=1)
    out = pd.DataFrame(
        {
            "family": [model_to_family[m] for m in s.index],
            "nearest": nn,
            "nearest_family": [model_to_family[m] for m in nn],
            "similarity": [sim.loc[m, n] for m, n in nn.items()],
        },
        index=s.index,
    )
    out["same_family"] = out["family"] == out["nearest_family"]
    return out.sort_values(["family", "similarity"], ascending=[True, False])


def plot_similarity_heatmap(
    sim: pd.DataFrame,
    model_to_family: dict,
    *,
    palette: dict | None = None,
    cluster: bool = True,
    method: str = "average",
    fmt: str = ".2f",
    cmap: str = "viridis",
    title: str | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] = (11, 10),
    show: bool = True,
):
    """Model x model similarity heatmap with a family colour strip.

    ``cluster=True`` reorders models by hierarchical clustering on
    ``1 - similarity`` (``method`` linkage) and draws the dendrogram;
    ``cluster=False`` keeps family order. The diagonal (always 1) is masked so
    the colour scale covers the off-diagonal range. Returns the heatmap Axes.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    palette = palette or make_palette(model_to_family.values())
    fam = pd.Series(model_to_family).loc[sim.index]
    row_colours = fam.map(palette).rename("family")
    mask = np.eye(len(sim), dtype=bool)
    off = sim.to_numpy()[~mask]

    if cluster:
        dist = 1 - sim.to_numpy()
        np.fill_diagonal(dist, 0)
        Z = linkage(squareform(dist, checks=False), method=method)
        g = sns.clustermap(
            sim,
            row_linkage=Z,
            col_linkage=Z,
            row_colors=row_colours,
            xticklabels=True,
            yticklabels=True,
            col_colors=row_colours,
            mask=mask,
            cmap=cmap,
            vmin=off.min(),
            vmax=off.max(),
            annot=True,
            fmt=fmt,
            annot_kws={"fontsize": 7},
            linewidths=0.5,
            linecolor="white",
            figsize=figsize,
            dendrogram_ratio=0.12,
            cbar_kws={"label": "mean cosine similarity"},
            cbar_pos=(1.04, 0.3, 0.02, 0.4),  # right of the row labels
        )
        ax, fig = g.ax_heatmap, g.figure
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title:
            g.figure.suptitle(title, y=1.02)
    else:
        order = fam.sort_values(kind="stable").index
        s = sim.loc[order, order]
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            s,
            mask=np.eye(len(s), dtype=bool),
            cmap=cmap,
            vmin=off.min(),
            vmax=off.max(),
            annot=True,
            fmt=fmt,
            annot_kws={"fontsize": 7},
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"label": "mean cosine similarity", "shrink": 0.7},
            ax=ax,
        )
        # Outline the family blocks
        sizes = fam.loc[order].value_counts(sort=False).reindex(fam.loc[order].unique())
        start = 0
        for n in sizes:
            ax.add_patch(plt.Rectangle((start, start), n, n, fill=False, ec="red", lw=1.5))
            start += n
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title:
            ax.set_title(title)
        fig.tight_layout()

    for lab in ax.get_yticklabels() + ax.get_xticklabels():
        lab.set_color(palette.get(model_to_family.get(lab.get_text()), "black"))
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ax
