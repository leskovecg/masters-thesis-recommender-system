"""
Figures for the defense slides (and the rating histogram of the thesis), from the exports of
the last run. Slovene labels, decimal comma, 300 dpi, same style as results_overview.

Run from the project root: python presentation/make_slide_figures.py
Output: presentation/figures/*.png and *.pdf; ratings_distribution (thesis figure) goes to
outputs/latex/figs/.
"""
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = "20261125_222631"
ROOT = Path(__file__).resolve().parents[1]
RECS = ROOT / "outputs" / "recommendations"
OUT = ROOT / "presentation" / "figures"
THESIS_FIGS = ROOT / "outputs" / "latex" / "figs"

BLUE, GREY, LIGHT_GREY = "#2A6FDB", "#9aa3b0", "#cfd4db"
MUTED, GRID = "#5b6474", "#e6e8ec"
ACTIVITY_QS = ["Ac_AB4_1", "Ac_AB4_2", "Ac_AB4_3", "Ac_AB4_4", "Ac_AB4_5", "Ac_AB4_8"]


def comma(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def thousands(value):
    return f"{int(value):,}".replace(",", " ")


def style(ax, title, grid_axis=None):
    ax.set_title(title, fontsize=12, loc="left", fontweight="bold")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LIGHT_GREY)
    ax.tick_params(length=0, colors=MUTED, labelsize=10)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7)
        ax.set_axisbelow(True)


def save(fig, name, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ratings_distribution():
    """All ratings, and the number of distinct ratings per user (the cause of ties)."""
    with open(ROOT / "data" / "D_lst" / "D_lst_full_20260202_150312.pkl", "rb") as f:
        D_lst = pickle.load(f)
    ratings = np.array([row[3] for row in D_lst])
    distinct = defaultdict(set)
    for uid, _, _, rating in D_lst:
        distinct[uid].add(round(rating, 9))
    counts = pd.Series([len(v) for v in distinct.values()]).value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.7), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.hist(ratings, bins=np.arange(0.3, 5.0001, 0.1), color=BLUE, rwidth=0.88)
    style(ax1, "Porazdelitev izračunanih ocen", "y")
    ax1.set_xlim(0.2, 5.1)
    ax1.set_xlabel("Ocena", color=MUTED)
    ax1.set_ylabel("Število ocen", color=MUTED)
    ax1.yaxis.set_major_formatter(lambda v, _: thousands(v))

    ax2.bar(counts.index, counts.values, width=0.62, color=BLUE)
    for x, v in counts.items():
        ax2.text(x, v + counts.max() * 0.02, v, ha="center", fontsize=10)
    style(ax2, "Različne ocene uporabnika")
    ax2.set_ylim(0, counts.max() * 1.12)
    ax2.set_xticks(range(counts.index.min(), counts.index.max() + 1))
    ax2.set_yticks([])
    ax2.set_xlabel("Število različnih ocen", color=MUTED)
    ax2.set_ylabel("Število uporabnikov", color=MUTED)
    fig.tight_layout(w_pad=2)
    save(fig, "ratings_distribution", THESIS_FIGS)


def context_reference():
    """679 users by number of feasible test items, and F1 where the choice is not trivial."""
    df = pd.read_excel(RECS / f"context_breakdown_{RUN}.xlsx").set_index("FeasibleTestItems")
    none = int(df["NumUsers"].iloc[0] + df["NumUsers"].iloc[1])
    few, many = int(df.at["1–5", "NumUsers"]), int(df.at["več kot 5", "NumUsers"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    total, left = none + few + many, 0
    for n, color, label in ((none, LIGHT_GREY, "0\nbrez\npriporočila"), (few, GREY, "1–5\nF1 = 1 vedno"),
                            (many, BLUE, "več kot 5\nizbor ni trivialen")):
        ax1.barh(0, n, left=left, height=0.5, color=color, edgecolor="white", linewidth=2)
        ax1.text(left + n / 2, 0, n, ha="center", va="center", fontsize=13, fontweight="bold",
                 color="white" if color == BLUE else "black")
        ax1.text(left + n / 2, -0.36, label, ha="center", va="top", fontsize=9.5, color=MUTED)
        left += n
    style(ax1, f"{total} uporabnikov po številu skladnih testnih kombinacij")
    ax1.set_ylim(-0.9, 0.45)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.spines["bottom"].set_visible(False)

    row = df.loc["več kot 5"]
    bars = [("SVD", row["F1_SVD"], BLUE), ("Popularnost", row["F1_Popularity"], GREY), ("Naključno", row["F1_Random"], GREY)]
    for y, (_, value, color) in zip((2, 1, 0), bars):
        ax2.barh(y, value, height=0.62, color=color)
        ax2.text(value + 0.015, y, comma(value), va="center", fontsize=11)
    style(ax2, f"F1 pri {many} uporabnikih (kontekstna ref.)")
    ax2.set_yticks([2, 1, 0], [b[0] for b in bars], fontsize=11, color="black")
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,5", "0,75", "1"])
    ax2.grid(axis="x", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    save(fig, "context_reference")


def group_results():
    """User-weighted F1 of M3-M5 and the distribution of the 34 used activities over the groups."""
    results = pd.read_excel(RECS / f"results_all_layers_{RUN}.xlsx").set_index("Category")
    f1 = {layer: results.at[f"{layer} (uteženo po uporabnikih)", "F1"] for layer in ("M3", "M4", "M5")}
    actions = pd.read_excel(ROOT / "data" / "context" / "ActivityContextGen_v12.xlsx", sheet_name="ActionLst")
    used = actions[actions["qID"].ffill().isin(ACTIVITY_QS)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    for y, (layer, value) in zip((2, 1, 0), f1.items()):
        ax1.barh(y, value, height=0.62, color=BLUE)
        ax1.text(value + 0.015, y, comma(value), va="center", fontsize=11)
        sizes = used[f"ContextG_act_{layer[1]}"].value_counts().sort_index()
        left = 0
        for g, n in sizes.items():
            ax2.barh(y, n, left=left, height=0.62, color=BLUE, edgecolor="white", linewidth=2.5)
            ax2.text(left + n / 2, y, n, ha="center", va="center", fontsize=11, color="white")
            ax2.text(left + n / 2, y - 0.4, g, ha="center", va="center", fontsize=8.5, color=MUTED)
            left += n
    style(ax1, "F1 (uteženo po uporabnikih)", "x")
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,5", "0,75", "1"])
    style(ax2, "Število uporabljenih aktivnosti po skupinah (skupaj 34)")
    ax2.set_xticks([])
    ax2.spines["bottom"].set_visible(False)
    ax2.set_xlabel("pod stolpcem: oznaka skupine", color=MUTED, fontsize=9.5)
    for ax in (ax1, ax2):
        ax.set_yticks([2, 1, 0], list(f1), fontsize=11, color="black")
        ax.set_ylim(-0.6, 2.5)
    fig.tight_layout(w_pad=2)
    save(fig, "group_results")


if __name__ == "__main__":
    ratings_distribution()
    context_reference()
    group_results()
    print(f"Saved figures to: {OUT} and {THESIS_FIGS}")
