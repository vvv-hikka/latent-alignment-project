"""Summaries + plots for the latent (CCS) and behavior (free-generation) experiments.

Behavior is now *free generation* labeled by an LLM judge into three classes
(``safe`` / ``harmful`` / ``gibberish``) — see ``latent_alignment.judge``. There is no more
regex ``stance`` column. ``label`` in the behavior CSVs is the statement's polarity:
``0`` = harmful statement, ``1`` = its benign negation (paired by ``pair_id``).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent

# (display name, ccs dir, behavior dir or None)
MODELS = [
    ("olmo2 base", "olmo2_1b_base_mixed", "behavior_olmo2_1b_base"),
    ("olmo2 instruct", "olmo2_1b_instruct_mixed", "behavior_olmo2_1b_it"),
    ("gemma3 base", "gemma3_1b_base_mixed", None),
    ("gemma3 instruct", "gemma3_1b_instruct_mixed", None),
    ("olmo1 base", "olmo_1b_mixed", "behavior_olmo_1b"),
]

JUDGE_LABELS = ["safe", "harmful", "gibberish"]
JUDGE_COLORS = {"safe": "#2ca02c", "harmful": "#d62728", "gibberish": "#9e9e9e"}


def load_ccs(d: str) -> pd.DataFrame | None:
    path = RUNS / d / "ccs_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["acc_corrected"] = np.maximum(df["accuracy"], 1 - df["accuracy"])
    return df


def load_behavior(d: str | None) -> pd.DataFrame | None:
    if d is None:
        return None
    path = RUNS / d / "behavior_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def ccs_row(name: str, df: pd.DataFrame) -> dict:
    best = df.loc[df["acc_corrected"].idxmax()]
    return {
        "model": name,
        "n_layers": len(df),
        "best_acc": round(float(best["acc_corrected"]), 3),
        "best_layer": int(best["layer"]),
        "mean_acc": round(float(df["acc_corrected"].mean()), 3),
        "max_silhouette": round(float(df["silhouette"].max()), 3),
        "mean_contradiction_idx": round(float(df["contradiction_index_mean"].mean()), 3),
        "polar_consistency_absmean": round(float(df["polar_consistency_mean"].abs().mean()), 3),
    }


def _harmful_rate(s: pd.Series) -> float:
    return float((s == "harmful").mean())


def behavior_row(name: str, df: pd.DataFrame) -> dict:
    """Free-generation metrics from the judge's safe/harmful/gibberish labels.

    ``harmful_rate_coherent`` excludes gibberish, so models that mostly emit noise (e.g. tiny
    base models) are compared on the answers that actually mean something.
    """
    n = len(df)
    rate = df["judge_label"].value_counts(normalize=True)
    safe = float(rate.get("safe", 0.0))
    harmful = float(rate.get("harmful", 0.0))
    gibberish = float(rate.get("gibberish", 0.0))
    coherent = safe + harmful
    by_label = df.groupby("label")["judge_label"]
    return {
        "model": name,
        "n": n,
        "safe_rate": round(safe, 3),
        "harmful_rate": round(harmful, 3),
        "gibberish_rate": round(gibberish, 3),
        "coherent_rate": round(coherent, 3),
        "harmful_rate_coherent": round(harmful / coherent, 3) if coherent else float("nan"),
        "harmful_l0_stmt": round(_harmful_rate(by_label.get_group(0)), 3) if 0 in df["label"].values else float("nan"),
        "harmful_l1_neg": round(_harmful_rate(by_label.get_group(1)), 3) if 1 in df["label"].values else float("nan"),
    }


def main() -> None:
    ccs = {name: load_ccs(d) for name, d, _ in MODELS}
    beh = {name: load_behavior(b) for name, _, b in MODELS}

    ccs_summary = pd.DataFrame([ccs_row(n, df) for n, df in ccs.items() if df is not None])
    beh_summary = pd.DataFrame([behavior_row(n, df) for n, df in beh.items() if df is not None])

    ccs_summary.to_csv(RUNS / "summary_ccs.csv", index=False)
    beh_summary.to_csv(RUNS / "summary_behavior.csv", index=False)
    print("=== summary_ccs.csv ===")
    print(ccs_summary.to_string(index=False))
    print("\n=== summary_behavior.csv ===")
    print(beh_summary.to_string(index=False))

    _plot_ccs(ccs)
    _plot_behavior(beh_summary, ccs_summary)
    _plot_latent_vs_behavior(ccs_summary, beh_summary)

    print("\nwrote: summary_ccs.csv, summary_behavior.csv, "
          "plot_ccs_accuracy.png, plot_behavior.png, plot_latent_vs_behavior.png")


def _plot_ccs(ccs: dict) -> None:
    """Layerwise corrected CCS accuracy — how well the latent separates harmful vs benign."""
    styles = [
        {"color": "#1f77b4", "ls": "-", "marker": "o"},
        {"color": "#ff7f0e", "ls": "--", "marker": "s"},
        {"color": "#2ca02c", "ls": "-.", "marker": "^"},
        {"color": "#d62728", "ls": ":", "marker": "D"},
        {"color": "#9467bd", "ls": "-", "marker": "v"},
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, df), st in zip(((n, d) for n, d in ccs.items() if d is not None), styles):
        ax.plot(df["layer"], df["acc_corrected"], ms=5, lw=1.8, alpha=0.9, label=name, **st)
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    ax.set(xlabel="layer", ylabel="corrected accuracy",
           title="Latent separability (CCS) by layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUNS / "plot_ccs_accuracy.png", dpi=130)
    plt.close(fig)


def _plot_behavior(beh_summary: pd.DataFrame, ccs_summary: pd.DataFrame) -> None:
    """Two panels: judge-label mix per model, and harmful rate split by statement polarity.

    The right panel also overlays the latent CCS accuracy (right axis) so "the model knows"
    (latent) sits next to "the model does" (harmful generation).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(beh_summary))

    # (a) stacked safe/harmful/gibberish proportions.
    bottom = np.zeros(len(beh_summary))
    for cat in JUDGE_LABELS:
        vals = beh_summary[f"{cat}_rate"].to_numpy()
        ax1.bar(x, vals, bottom=bottom, color=JUDGE_COLORS[cat], label=cat)
        bottom += vals
    ax1.set_xticks(x)
    ax1.set_xticklabels(beh_summary["model"], rotation=15, ha="right", fontsize=8)
    ax1.set(ylabel="proportion of generations", ylim=(0, 1),
            title="Free-generation outcome (LLM judge)")
    ax1.legend(fontsize=8)

    # (b) harmful rate on harmful statements vs their benign negations.
    w = 0.38
    ax2.bar(x - w / 2, beh_summary["harmful_l0_stmt"], w, color="#d62728",
            label="prompted with the hateful statement")
    ax2.bar(x + w / 2, beh_summary["harmful_l1_neg"], w, color="#f4a3a3",
            label="prompted with its negated (benign) version")
    ax2.set_xticks(x)
    ax2.set_xticklabels(beh_summary["model"], rotation=15, ha="right", fontsize=8)
    ax2.set(ylabel="rate of harmful generations",
            title="PA-CCS accuracy (line) vs harmful output it generates (bars)")

    # Overlay latent CCS accuracy on a second axis ("the model knows it's harmful").
    acc = beh_summary.merge(ccs_summary[["model", "best_acc"]], on="model", how="left")["best_acc"]
    ax3 = ax2.twinx()
    ax3.plot(x, acc, "k--D", ms=7, lw=1.6, label="latent accuracy (PA-CCS)")
    ax3.axhline(0.5, ls=":", c="gray", lw=1)
    ax3.set_ylim(0.45, 1.0)
    ax3.set_ylabel("latent accuracy (PA-CCS)")

    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2 + h3, l2 + l3, fontsize=8, loc="upper center")

    fig.tight_layout()
    fig.savefig(RUNS / "plot_behavior.png", dpi=130)
    plt.close(fig)


def _plot_latent_vs_behavior(ccs_summary: pd.DataFrame, beh_summary: pd.DataFrame) -> None:
    """The dissociation: latent separability (x) vs harmful free-generation rate (y)."""
    merged = ccs_summary.merge(beh_summary, on="model", how="inner")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(merged["best_acc"], merged["harmful_rate_coherent"], s=70, color="#d62728")
    for _, r in merged.iterrows():
        ax.annotate(f"{r['model']}\n(coherent {r['coherent_rate']:.0%})",
                    (r["best_acc"], r["harmful_rate_coherent"]),
                    fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set(xlabel="latent best accuracy (CCS)",
           ylabel="harmful rate among coherent generations",
           title="Latent 'knows' vs behavior 'does'")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RUNS / "plot_latent_vs_behavior.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
