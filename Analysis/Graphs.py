import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --- Load & clean ---
df = pd.read_csv("../data/profiling_results.csv")
df.replace(-1, np.nan, inplace=True)
df["label"] = df["name"] + " N=" + df["N"].astype(str)

# --- Style ---
plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "axes.edgecolor": "#313244",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#6c7086",
    "ytick.color": "#6c7086",
    "text.color": "#cdd6f4",
    "grid.color": "#313244",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
})

def tier_color(rate):
    if rate >= 55: return "#ef4444"
    if rate >= 15: return "#f59e0b"
    return "#22c55e"

# Tier legend patches (used in multiple graphs)
from matplotlib.patches import Patch
legend = [Patch(color="#ef4444", label="HIGH ≥55%"),
          Patch(color="#f59e0b", label="MEDIUM 15-54%"),
          Patch(color="#22c55e", label="LOW <15%")]

# ================================================================
# 1. Contradiction rate bar chart (sorted)
# ================================================================
fig, ax = plt.subplots(figsize=(16, 6))
df_sorted = df.sort_values("contradiction_rate", ascending=False)
colors = [tier_color(r) for r in df_sorted["contradiction_rate"]]
ax.bar(df_sorted["label"], df_sorted["contradiction_rate"], color=colors, edgecolor="#11111b", linewidth=0.5)
ax.set_title("Contradiction Rate by Sample", fontsize=14, pad=16)
ax.set_ylabel("Contradiction Rate (%)")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted["label"], rotation=45, ha="right", fontsize=8)
ax.grid(axis="y")
ax.legend(handles=legend, loc="upper right", framealpha=0.2)
plt.tight_layout()
plt.savefig("../data/graph_contradiction_rate.png", dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 2. Average time per attempt (sorted by time)
# ================================================================
fig, ax = plt.subplots(figsize=(16, 6))
df_time = df.sort_values("avg_ms", ascending=False)
ax.bar(df_time["label"], df_time["avg_ms"], color="#89b4fa", edgecolor="#11111b", linewidth=0.5, label="avg_ms")
ax.bar(df_time["label"], df_time["avg_ms_success"], color="#a6e3a1", edgecolor="#11111b", linewidth=0.5, alpha=0.7, label="avg_ms_success")
ax.bar(df_time["label"], df_time["avg_ms_contradiction"], color="#f38ba8", edgecolor="#11111b", linewidth=0.5, alpha=0.7, label="avg_ms_contradiction")
ax.set_title("Average Attempt Time by Sample", fontsize=14, pad=16)
ax.set_ylabel("Time (ms)")
ax.set_xticks(range(len(df_time)))
ax.set_xticklabels(df_time["label"], rotation=45, ha="right", fontsize=8)
ax.legend(framealpha=0.2)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("../data/graph_avg_time.png", dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 3. Contradiction rate vs average time (scatter)
# ================================================================
fig, ax = plt.subplots(figsize=(10, 7))
scatter_colors = [tier_color(r) for r in df["contradiction_rate"]]
ax.scatter(df["avg_ms"], df["contradiction_rate"], c=scatter_colors, s=80, edgecolors="#11111b", linewidth=0.5, zorder=3)
for _, row in df.iterrows():
    ax.annotate(row["label"], (row["avg_ms"], row["contradiction_rate"]),
                fontsize=7, color="#6c7086",
                xytext=(4, 4), textcoords="offset points")
ax.set_title("Contradiction Rate vs Average Attempt Time", fontsize=14, pad=16)
ax.set_xlabel("Average Time per Attempt (ms)")
ax.set_ylabel("Contradiction Rate (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True)
ax.legend(handles=legend, loc="upper right", framealpha=0.2)
plt.tight_layout()
plt.savefig("../data/graph_rate_vs_time.png", dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 4. Effect of N value on contradiction rate (box plot)
# ================================================================
fig, ax = plt.subplots(figsize=(8, 6))
n_groups = [df[df["N"] == n]["contradiction_rate"].dropna().values for n in sorted(df["N"].unique())]
n_labels = [f"N={n}" for n in sorted(df["N"].unique())]
bp = ax.boxplot(n_groups, labels=n_labels, patch_artist=True,
                medianprops=dict(color="#f38ba8", linewidth=2))
colors_box = ["#89b4fa", "#cba6f7", "#a6e3a1", "#fab387"]
for patch, color in zip(bp["boxes"], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Effect of N Value on Contradiction Rate", fontsize=14, pad=16)
ax.set_ylabel("Contradiction Rate (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("../data/graph_n_effect.png", dpi=150, bbox_inches="tight")
plt.show()

# ================================================================
# 5. Periodic vs non-periodic contradiction rate (bar)
# ================================================================
fig, ax = plt.subplots(figsize=(6, 5))
periodic_avg = df.groupby("periodic")["contradiction_rate"].mean()
colors_p = ["#cba6f7", "#89b4fa"]
bars = ax.bar(["Non-Periodic", "Periodic"],
              [periodic_avg.get(False, 0), periodic_avg.get(True, 0)],
              color=colors_p, edgecolor="#11111b", linewidth=0.5, width=0.5)
ax.set_title("Average Contradiction Rate:\nPeriodic vs Non-Periodic", fontsize=13, pad=16)
ax.set_ylabel("Average Contradiction Rate (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(axis="y")
for bar, val in zip(bars, [periodic_avg.get(False, 0), periodic_avg.get(True, 0)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=11, color="#cdd6f4")
plt.tight_layout()
plt.savefig("../data/graph_periodic_effect.png", dpi=150, bbox_inches="tight")
plt.show()

print("All graphs saved to ../data/")
