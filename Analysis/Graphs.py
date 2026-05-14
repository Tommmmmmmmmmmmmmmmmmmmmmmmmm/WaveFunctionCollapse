"""
WFC Backtracking Strategy Analysis
Primary metric: avg_observations (efficiency)
Secondary metric: contradiction_rate (reliability, brief)

Usage: python analyse.py
Output: figures saved to ./figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# --- UPDATE THESE TWO PATHS ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "Release", "data", "combined_profiling_results.csv")
OUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "wfc_figures")

if not os.path.isfile(DATA_PATH):
    print(f"ERROR: CSV not found at:\n  {os.path.abspath(DATA_PATH)}")
    print(f"\nUpdate DATA_PATH at the top of this script to point to your CSV.")
    exit(1)
os.makedirs(OUT_DIR, exist_ok=True)

SINGLE_COL = (3.5, 2.6)
DOUBLE_COL = (7.16, 3.4)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize":  7,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PAL = {
    "Baseline": "#222222",
    "Fixed":    "#0077BB",
    "Relative": "#CC3311",
}

# ---------------------------------------------------------------------------
# Load & prepare
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows  |  {df['name'].nunique()} unique samples")

MERGE_KEY = ["name", "filename", "symmetry", "heuristic"]
baseline = df[df.strategy == "Baseline"][MERGE_KEY + ["avg_observations"]].copy()
baseline = baseline.rename(columns={"avg_observations": "baseline_obs"})
df = df.merge(baseline, on=MERGE_KEY, how="left")

df["obs_ratio"] = df["avg_observations"] / df["baseline_obs"]

agg = (
    df.groupby(["strategy", "parameter"])
    .agg(
        mean_obs=("avg_observations", "mean"),
        median_obs=("avg_observations", "median"),
        mean_ratio=("obs_ratio", "mean"),
        median_ratio=("obs_ratio", "median"),
        mean_cr=("contradiction_rate", "mean"),
    )
    .reset_index()
)

baseline_agg = agg[agg.strategy == "Baseline"]
fixed_agg    = agg[agg.strategy == "Fixed"].sort_values("parameter")
relative_agg = agg[agg.strategy == "Relative"].sort_values("parameter")

best_fixed_k = fixed_agg.loc[fixed_agg.mean_ratio.idxmin(), "parameter"]
best_rel_f   = relative_agg.loc[relative_agg.mean_ratio.idxmin(), "parameter"]

print(f"Best Fixed k = {best_fixed_k}  |  Best Relative f = {best_rel_f}")

# =========================================================================
# FIGURE 1a — Fixed strategy obs ratio (own y-scale)
# =========================================================================

fig, ax = plt.subplots(figsize=SINGLE_COL)
ax.plot(
    fixed_agg.parameter, fixed_agg.mean_ratio,
    color=PAL["Fixed"], marker="s", markersize=4, linewidth=1.2,
    label="Mean obs. ratio",
)
ax.axhline(1.0, linestyle="--", linewidth=0.9, color=PAL["Baseline"],
           label="Baseline (restart)")
ax.fill_between(fixed_agg.parameter, 1.0, fixed_agg.mean_ratio,
                where=fixed_agg.mean_ratio < 1.0,
                alpha=0.15, color=PAL["Fixed"], interpolate=True)

# Mark best k
best_row = fixed_agg.loc[fixed_agg.mean_ratio.idxmin()]
ax.annotate(
    f"k={int(best_row.parameter)}\n{best_row.mean_ratio:.3f}",
    (best_row.parameter, best_row.mean_ratio),
    fontsize=6, ha="center",
    xytext=(0, -18), textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
)

ax.set_xlabel("Fixed backtrack distance k (steps)")
ax.set_ylabel("Avg observations (ratio to baseline)")
ax.set_title("Fixed strategy — efficiency vs baseline")
ax.set_xlim(left=0)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1a_fixed_obs_ratio.png")
fig.savefig(f"{OUT_DIR}/fig1a_fixed_obs_ratio.pdf")
print("Saved fig1a_fixed_obs_ratio")

# =========================================================================
# FIGURE 1b — Relative strategy obs ratio (own y-scale)
# =========================================================================

fig, ax = plt.subplots(figsize=SINGLE_COL)
ax.plot(
    relative_agg.parameter, relative_agg.mean_ratio,
    color=PAL["Relative"], marker="^", markersize=4, linewidth=1.2,
    label="Mean obs. ratio",
)
ax.axhline(1.0, linestyle="--", linewidth=0.9, color=PAL["Baseline"],
           label="Baseline (restart)")
ax.fill_between(relative_agg.parameter, 1.0, relative_agg.mean_ratio,
                where=relative_agg.mean_ratio < 1.0,
                alpha=0.15, color=PAL["Relative"], interpolate=True)
ax.fill_between(relative_agg.parameter, 1.0, relative_agg.mean_ratio,
                where=relative_agg.mean_ratio > 1.0,
                alpha=0.10, color="red", interpolate=True)

best_r_row = relative_agg.loc[relative_agg.mean_ratio.idxmin()]
ax.annotate(
    f"f={best_r_row.parameter}\n{best_r_row.mean_ratio:.3f}",
    (best_r_row.parameter, best_r_row.mean_ratio),
    fontsize=6, ha="center",
    xytext=(0, -18), textcoords="offset points",
    arrowprops=dict(arrowstyle="-", color="grey", lw=0.6),
)

ax.set_xlabel("Relative backtrack factor f")
ax.set_ylabel("Avg observations (ratio to baseline)")
ax.set_title("Relative strategy — efficiency vs baseline")
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1b_relative_obs_ratio.png")
fig.savefig(f"{OUT_DIR}/fig1b_relative_obs_ratio.pdf")
print("Saved fig1b_relative_obs_ratio")

# =========================================================================
# FIGURE 2 — Per-sample obs ratio distribution (boxplot at best params)
# =========================================================================

conditions = [
    ("Baseline",                     df.strategy == "Baseline"),
    (f"Fixed k={int(best_fixed_k)}", (df.strategy == "Fixed") & (df.parameter == best_fixed_k)),
    (f"Relative f={best_rel_f}",     (df.strategy == "Relative") & (df.parameter == best_rel_f)),
]

fig, ax = plt.subplots(figsize=(3.5, 3.0))
data_to_plot = [df.loc[mask, "obs_ratio"].values for _, mask in conditions]
labels = [label for label, _ in conditions]

bp = ax.boxplot(
    data_to_plot,
    tick_labels=labels,
    patch_artist=True,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
    flierprops=dict(marker=".", markersize=3, alpha=0.5),
    widths=0.5,
)
for patch, c in zip(bp["boxes"], ["#AAAAAA", PAL["Fixed"], PAL["Relative"]]):
    patch.set_facecolor(c)
    patch.set_alpha(0.55)

ax.axhline(1.0, linestyle="--", linewidth=0.7, color="grey", alpha=0.7)
ax.set_ylabel("Observation ratio (vs baseline)")
ax.set_title("Per-sample efficiency at best parameter")
ax.tick_params(axis="x", labelrotation=8)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_obs_distribution.png")
fig.savefig(f"{OUT_DIR}/fig2_obs_distribution.pdf")
print("Saved fig2_obs_distribution")

# =========================================================================
# FIGURE 3 — Scatter: difficulty vs improvement
#   Version A: heavily annotated
#   Version B: clean (no labels)
# =========================================================================

best_fixed_df = df[(df.strategy == "Fixed") & (df.parameter == best_fixed_k)].copy()
best_rel_df   = df[(df.strategy == "Relative") & (df.parameter == best_rel_f)].copy()


def make_fig3(annotate, suffix, datasets, palette, figsize, out_dir):
    """
    datasets: list of (dataframe, colour_key, title_str) tuples, one per subplot.
    """
    n = len(datasets)
    fig3, axes3 = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes3 = [axes3]

    for ax3, (data, colour_key, panel_title) in zip(axes3, datasets):
        ax3.scatter(
            data.baseline_obs, data.obs_ratio,
            s=18, alpha=0.6, color=palette[colour_key],
            edgecolors="white", linewidths=0.3,
        )
        ax3.axhline(1.0, linestyle="--", linewidth=0.7, color="grey")
        ax3.set_xlabel("Baseline avg observations (sample difficulty)")
        ax3.set_title(panel_title)

        if annotate:
            to_label = data[(data.obs_ratio < 0.95) | (data.obs_ratio > 1.05)].copy()
            extremes = pd.concat([data.nsmallest(5, "obs_ratio"),
                                  data.nlargest(5, "obs_ratio")]).drop_duplicates()
            to_label = pd.concat([to_label, extremes]).drop_duplicates()

            used_positions = []
            for _, row in to_label.iterrows():
                pt_x, pt_y = row.baseline_obs, row.obs_ratio
                oy = 6 if len(used_positions) % 2 == 0 else -10
                ax3.annotate(
                    row["name"], (pt_x, pt_y),
                    fontsize=5, alpha=0.85,
                    xytext=(5, oy), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4),
                )
                used_positions.append((pt_x, pt_y))

    axes3[0].set_ylabel("Observation ratio (vs baseline)")
    fig3.suptitle("Does sample difficulty predict backtracking benefit?", y=1.02)
    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_difficulty_vs_benefit_{suffix}.png")
    fig3.savefig(f"{out_dir}/fig3_difficulty_vs_benefit_{suffix}.pdf")
    print(f"Saved fig3_difficulty_vs_benefit_{suffix}")


fig3_datasets = [
    (best_fixed_df, "Fixed",    f"(a) Fixed k={int(best_fixed_k)}"),
    (best_rel_df,   "Relative", f"(b) Relative f={best_rel_f}"),
]

make_fig3(True,  "labelled", fig3_datasets, PAL, DOUBLE_COL, OUT_DIR)
make_fig3(False, "clean",    fig3_datasets, PAL, DOUBLE_COL, OUT_DIR)

# =========================================================================
# FIGURE 4 — Per-sample lines across Relative parameter sweep
# =========================================================================

rel_df = df[df.strategy == "Relative"].copy()
sample_var = (
    rel_df.groupby("name")["obs_ratio"]
    .agg(["min", "max", "std"])
    .assign(range=lambda d: d["max"] - d["min"])
    .sort_values("range", ascending=False)
)
top_varied = sample_var.head(8).index.tolist()

fig, ax = plt.subplots(figsize=DOUBLE_COL)
cmap = plt.get_cmap("tab10")

for i, name in enumerate(top_varied):
    sub = (
        rel_df[rel_df.name == name]
        .groupby("parameter")["obs_ratio"]
        .mean()
        .reset_index()
        .sort_values("parameter")
    )
    ax.plot(
        sub.parameter, sub.obs_ratio,
        marker="o", markersize=3.5, linewidth=1.1,
        label=name, color=cmap(i),
    )

ax.axhline(1.0, linestyle="--", linewidth=0.7, color="grey", label="Baseline")
ax.set_xlabel("Relative backtrack factor f")
ax.set_ylabel("Observation ratio (vs baseline)")
ax.set_title("Relative strategy — samples with greatest variation in efficiency")
ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
ax.legend(loc="upper left", ncol=2, fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_relative_per_sample.png")
fig.savefig(f"{OUT_DIR}/fig4_relative_per_sample.pdf")
print("Saved fig4_relative_per_sample")

# =========================================================================
# FIGURE 5 — Contradiction rate (secondary, compact)
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=DOUBLE_COL, sharey=True)

ax = axes[0]
ax.bar(fixed_agg.parameter, fixed_agg.mean_cr,
       width=80, color=PAL["Fixed"], alpha=0.7, edgecolor="none")
ax.set_xlabel("Fixed backtrack distance k")
ax.set_ylabel("Mean contradiction rate (%)")
ax.set_title("(a) Fixed strategy")
ax.set_xlim(left=0)

ax2 = axes[1]
ax2.bar(relative_agg.parameter, relative_agg.mean_cr,
        width=0.07, color=PAL["Relative"], alpha=0.7, edgecolor="none")
ax2.set_xlabel("Relative backtrack factor f")
ax2.set_title("(b) Relative strategy")
ax2.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

fig.suptitle("Contradiction rate by parameter (secondary metric)", y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_contradiction_rate.png")
fig.savefig(f"{OUT_DIR}/fig5_contradiction_rate.pdf")
print("Saved fig5_contradiction_rate")

# =========================================================================
# Console summary
# =========================================================================

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

bl_obs = baseline_agg.mean_obs.values[0]
print(f"\nBaseline mean observations: {bl_obs:.1f}")

best_f = fixed_agg.loc[fixed_agg.mean_ratio.idxmin()]
print(f"\nFixed — best k = {int(best_f.parameter)}")
print(f"  Mean ratio: {best_f.mean_ratio:.4f}  ({(best_f.mean_ratio - 1)*100:+.1f}%)")

best_r = relative_agg.loc[relative_agg.mean_ratio.idxmin()]
print(f"\nRelative — best f = {best_r.parameter}")
print(f"  Mean ratio: {best_r.mean_ratio:.4f}  ({(best_r.mean_ratio - 1)*100:+.1f}%)")

for label, sub_df in [
    (f"Fixed k={int(best_fixed_k)}", df[(df.strategy == "Fixed") & (df.parameter == best_fixed_k)]),
    (f"Relative f={best_rel_f}",     df[(df.strategy == "Relative") & (df.parameter == best_rel_f)]),
]:
    n_better = (sub_df.obs_ratio < 0.99).sum()
    n_worse  = (sub_df.obs_ratio > 1.01).sum()
    n_same   = len(sub_df) - n_better - n_worse
    print(f"{label}: {n_better} improved, {n_worse} worsened, {n_same} ~unchanged")

print(f"\nAll figures saved to {OUT_DIR}/")
plt.show()

# --- Stats verification ---
fixed_best = df[(df.strategy == "Fixed") & (df.parameter == best_fixed_k)]
rel_best   = df[(df.strategy == "Relative") & (df.parameter == best_rel_f)]

# =====================================================================
print("=" * 65)
print("RESULTS DRAFT — STATISTICAL CLAIMS")
print("=" * 65)

# --- Header ---
n_configs = len(df[df.strategy == "Baseline"])
print(f"\nTotal sample configurations: {n_configs}")
print(f"Attempts per configuration:  {df['attempts'].iloc[0]}")

# --- Fixed-Distance Backtracking ---
print("\n" + "-" * 65)
print("FIXED-DISTANCE BACKTRACKING")
print("-" * 65)

best_fixed_row = fixed_agg.loc[fixed_agg.mean_ratio.idxmin()]
print(f"\nBest k:          {int(best_fixed_k)}")
print(f"Mean obs ratio:  {best_fixed_row.mean_ratio:.3f}")
print(f"Reduction:       {(1 - best_fixed_row.mean_ratio) * 100:.1f}%")

n_improved  = (fixed_best.obs_ratio < 0.99).sum()
n_unchanged = ((fixed_best.obs_ratio >= 0.99) & (fixed_best.obs_ratio <= 1.01)).sum()
n_worsened  = (fixed_best.obs_ratio > 1.01).sum()
print(f"\nAt k={int(best_fixed_k)}:")
print(f"  Improved (<0.99):   {n_improved} of {len(fixed_best)}")
print(f"  Unchanged:          {n_unchanged} of {len(fixed_best)}")
print(f"  Worsened (>1.01):   {n_worsened} of {len(fixed_best)}")

median_ratio = fixed_best.obs_ratio.median()
print(f"  Median obs ratio:   {median_ratio:.3f}")

most_improved = fixed_best.loc[fixed_best.obs_ratio.idxmin()]
most_worsened = fixed_best.loc[fixed_best.obs_ratio.idxmax()]
print(f"\n  Most improved:  {most_improved['name']} "
      f"(N={most_improved['N']}, sym={most_improved['symmetry']}) "
      f"ratio={most_improved.obs_ratio:.3f} "
      f"({(1 - most_improved.obs_ratio) * 100:.1f}% reduction)")
print(f"  Most worsened:  {most_worsened['name']} "
      f"(N={most_worsened['N']}, sym={most_worsened['symmetry']}) "
      f"ratio={most_worsened.obs_ratio:.3f}")

# --- Progress-Relative Backtracking ---
print("\n" + "-" * 65)
print("PROGRESS-RELATIVE BACKTRACKING")
print("-" * 65)

best_rel_row = relative_agg.loc[relative_agg.mean_ratio.idxmin()]
print(f"\nBest f:          {best_rel_f}")
print(f"Mean obs ratio:  {best_rel_row.mean_ratio:.3f}")
print(f"Increase:        {(best_rel_row.mean_ratio - 1) * 100:.1f}%")

# U-shape extremes
for f_val in [0.1, 0.9]:
    row = relative_agg[relative_agg.parameter == f_val].iloc[0]
    print(f"  f={f_val}: mean ratio = {row.mean_ratio:.3f}")

# Variance comparison
fixed_std  = fixed_best.obs_ratio.std()
rel_std    = rel_best.obs_ratio.std()
print(f"\nStd dev at best params:")
print(f"  Fixed  k={int(best_fixed_k)}: {fixed_std:.3f}")
print(f"  Relative f={best_rel_f}:  {rel_std:.3f}")

# Per-sample extremes
rel_most_improved = rel_best.loc[rel_best.obs_ratio.idxmin()]
rel_most_worsened = rel_best.loc[rel_best.obs_ratio.idxmax()]
print(f"\n  Most improved:  {rel_most_improved['name']} "
      f"(N={rel_most_improved['N']}) "
      f"ratio={rel_most_improved.obs_ratio:.3f}")
print(f"  Most worsened:  {rel_most_worsened['name']} "
      f"(N={rel_most_worsened['N']}) "
      f"ratio={rel_most_worsened.obs_ratio:.3f}")

# --- Contradiction Rate ---
print("\n" + "-" * 65)
print("CONTRADICTION RATE")
print("-" * 65)

# Fixed
fixed_cr_peak = fixed_agg.loc[fixed_agg.mean_cr.idxmax()]
fixed_cr_zero_k = fixed_agg[fixed_agg.mean_cr == 0].parameter.min()
fixed_affected = df[(df.strategy == "Fixed") & (df.contradiction_rate > 0)].name.nunique()
print(f"\nFixed strategy:")
print(f"  Peak mean CR:       {fixed_cr_peak.mean_cr:.2f}% at k={int(fixed_cr_peak.parameter)}")
print(f"  CR drops to 0 at:   k >= {int(fixed_cr_zero_k)}")
print(f"  Samples ever affected: {fixed_affected}")

# Relative
rel_affected = df[(df.strategy == "Relative") & (df.contradiction_rate > 0)].name.nunique()
rel_max_cr = df[df.strategy == "Relative"].contradiction_rate.max()
rel_max_cr_row = df[(df.strategy == "Relative") & (df.contradiction_rate == rel_max_cr)].iloc[0]
print(f"\nRelative strategy:")
print(f"  Samples ever affected: {rel_affected}")
print(f"  Peak individual CR:    {rel_max_cr:.0f}% "
      f"({rel_max_cr_row['name']}, N={rel_max_cr_row['N']}, f={rel_max_cr_row.parameter})")

for f_val in [0.1, 0.9]:
    row = relative_agg[relative_agg.parameter == f_val].iloc[0]
    print(f"  f={f_val}: mean CR = {row.mean_cr:.2f}%")

# --- Search depth range (mentioned in Analysis) ---
print("\n" + "-" * 65)
print("ANALYSIS SUPPORT")
print("-" * 65)

bl = df[df.strategy == "Baseline"]
print(f"\nBaseline search depth range:")
print(f"  Min avg observations: {bl.avg_observations.min():.0f} "
      f"({bl.loc[bl.avg_observations.idxmin(), 'name']})")
print(f"  Max avg observations: {bl.avg_observations.max():.0f} "
      f"({bl.loc[bl.avg_observations.idxmax(), 'name']})")

print("\n" + "=" * 65)

from scipy.stats import wilcoxon
stat, p = wilcoxon(fixed_best.baseline_obs, fixed_best.avg_observations)
print(f"Wilcoxon p = {p:.4f} and stat = {stat}")