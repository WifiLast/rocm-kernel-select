import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# --- data: op / dtype label -> speedup multiplier (patched vs stock) ---
data = [
    ("conv1d depthwise · fp16", 16.81),
    ("conv1d depthwise · fp32", 9.24),
    ("conv3d · fp16", 4.72),
    ("conv3d · fp32", 2.48),
    ("linear · fp16", 2.45),
    ("linear · bf16", 2.41),
    ("sdpa (causal) · fp16", 1.03),
    ("linear · fp32", 1.03),
    ("group_norm · fp16", 1.02),
    ("conv2d · fp32", 1.01),
    ("matmul · fp32", 1.00),
    ("matmul · fp16", 0.99),
    ("conv2d · fp16", 0.99),
    ("group_norm · fp32", 0.95),
    ("bmm · fp32", 0.93),
    ("bmm · fp16", 0.78),
]
data.sort(key=lambda d: d[1], reverse=True)
labels = [d[0] for d in data]
speedups = np.array([d[1] for d in data])
log_vals = np.log2(speedups)

# --- validated palette (see references/palette.md) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

n = len(labels)
fig_h = 0.44 * n + 2.0
fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

y = np.arange(n)[::-1]  # top row = biggest speedup
colors = [GOOD if v >= 1.0 else CRITICAL for v in speedups]

bar_h = 0.6
ax.barh(y, log_vals, height=bar_h, color=colors, zorder=3, edgecolor=SURFACE, linewidth=1.5)

# baseline at 1x
ax.axvline(0, color=BASELINE_AXIS, linewidth=1.2, zorder=2)

# gridlines at powers of two
tick_mults = [0.5, 1, 2, 4, 8, 16]
tick_logs = [np.log2(t) for t in tick_mults]
for tl in tick_logs:
    ax.axvline(tl, color=GRIDLINE, linewidth=0.8, zorder=1)

ax.set_xticks(tick_logs)
ax.set_xticklabels([f"{t:g}x" for t in tick_mults], color=INK_MUTED, fontsize=9.5)
ax.set_yticks(y)
ax.set_yticklabels(labels, color=INK_PRIMARY, fontsize=10)

# direct value labels at bar ends
for yi, lv, sp, c in zip(y, log_vals, speedups, colors):
    pad = 0.09 if lv >= 0 else -0.09
    ha = "left" if lv >= 0 else "right"
    ax.text(lv + pad, yi, f"{sp:.2f}x", va="center", ha=ha,
             color=c, fontsize=9.5, fontweight="bold", zorder=4)

xmin, xmax = np.log2(0.6), np.log2(22)
ax.set_xlim(xmin, xmax)
ax.set_ylim(-0.8, n - 0.2)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
ax.set_axisbelow(True)

ax.text(0.06, -0.65, "stock baseline (1x, no change)",
        ha="left", va="center", color=INK_SECONDARY, fontsize=8.5,
        style="italic")

# legend (status encoding: good vs regression)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=GOOD, edgecolor="none", label="Faster than stock"),
    Patch(facecolor=CRITICAL, edgecolor="none", label="Slower than stock"),
]
leg = ax.legend(handles=legend_handles, loc="lower right", frameon=False,
                 fontsize=9.5, labelcolor=INK_SECONDARY, handlelength=1.2,
                 handleheight=1.2, bbox_to_anchor=(1.0, -0.09))

fig.suptitle("amd_tuned_torch: patched vs. stock ROCm kernels",
             x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK_PRIMARY, y=0.985)
ax.set_title("Speedup by op / dtype — AMD Radeon RX 7900 XTX, ROCm 7.2, torch 2.15 dev (log scale)",
             loc="left", fontsize=10.5, color=INK_SECONDARY, pad=14)

fig.text(0.02, 0.005, "Source: tools/bench_fast.py — min over 3 rounds, same call site, patch toggled on/off.",
          fontsize=8.5, color=INK_MUTED, ha="left")

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig("articel/speedup_chart.png", facecolor=SURFACE, bbox_inches="tight")
print("saved articel/speedup_chart.png")
