"""Figures for the public results summary: how the PSMAReg ranking comes about."""
import csv, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

H = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
R = json.load(open(H / "results/ranking_final_submissions16_cap.json")); A = json.load(open(H / "results/ranking_final_all_cap.json"))
M = json.load(open(H / "docker_team_mapping.json"))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"; INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
teams = list(R["official"]); K = len(teams)
label = lambda t: "MIAgent" if t == "EXP130_v2" else t
METRICS = [("CT_DSC", "CT DSC"), ("PET_DSC", "PET DSC"), ("CT_HD95", "CT HD95"), ("PET_HD95", "PET HD95"), ("absMTV", "|ΔMTV|"), ("absTLG", "|ΔTLG|"), ("adjNDV", "%NDV")]

def strip(ax, left=True):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if not left: ax.spines["left"].set_visible(False)

# 1. per-metric significance scores -------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 6.2))
grid = np.array([[R["official"][t]["sig_" + m] for m, _ in METRICS] for t in teams])
im = ax.imshow(grid, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
for i in range(K):
    for j in range(len(METRICS)):
        v = grid[i, j]; ax.text(j, i, f"{int(round(v * K)) - 1}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([n for _, n in METRICS]); ax.xaxis.tick_top()
ax.set_yticks(range(K)); ax.set_yticklabels([f"{i + 1}. {label(t)}" for i, t in enumerate(teams)])
ax.tick_params(length=0); strip(ax); ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
for x in (3.5, 5.5): ax.axvline(x, color="white", lw=3)
ax.text(1.5, -1.6, "Accuracy (weight 0.4)", ha="center", color=INK2, fontsize=9); ax.text(4.5, -1.6, "Biomarker (0.4)", ha="center", color=INK2, fontsize=9); ax.text(6, -1.6, "Regularity (0.2)", ha="center", color=INK2, fontsize=9)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("significance score = (methods beaten + 1) / 16", color=INK2); cb.outline.set_visible(False)
ax.set_title("Cell value: number of the other 15 methods significantly outperformed on that metric\n(one-sided Wilcoxon on 80 patient-level values, Holm-adjusted); rows in final-rank order", fontsize=9, color=INK2, loc="left", pad=40)
fig.tight_layout(); fig.savefig(OUT / "fig_metric_scores.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 2. components -> final -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.6))
y = np.arange(K)[::-1]; h = 0.26
comps = [("comp_accuracy", "Accuracy", BLUE), ("comp_biomarker", "Biomarker preservation (capped at accuracy)", ORANGE), ("comp_regularity", "Regularity (capped at accuracy)", AQUA)]
for k, (key, name, col) in enumerate(comps):
    vals = [R["official"][t][key] for t in teams]
    ax.barh(y + (1 - k) * h, vals, height=h * 0.92, color=col, label=name, edgecolor="#fcfcfb", linewidth=0.8)
for i, t in enumerate(teams):
    ax.text(1.02, y[i], f"{R['official'][t]['final']:.1f}", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.text(1.02, K - 0.3, "Final", fontsize=8.5, color=INK2, va="bottom")
ax.set_yticks(y); ax.set_yticklabels([f"{i + 1}. {label(t)}" for i, t in enumerate(teams)]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1.0); ax.set_xlabel("component score (mean significance score of its metrics)"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
strip(ax, left=False); ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Final = 100 × accuracy^0.4 × biomarker^0.4 × regularity^0.2", loc="left", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_components.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 3. rank stability ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for i, t in enumerate(teams):
    lo, hi = R["bootstrap"][t]["ci"]; med = R["bootstrap"][t]["median_rank"]
    ax.plot([lo, hi], [y[i], y[i]], color=BLUE, lw=2, solid_capstyle="round", alpha=0.45)
    ax.plot(med, y[i], "o", color="#fcfcfb", mec=BLUE, mew=1.6, ms=7)
    ax.plot(i + 1, y[i], "D", color=ORANGE, ms=6)
ax.set_yticks(y); ax.set_yticklabels([label(t) for t in teams]); ax.tick_params(axis="y", length=0)
ax.set_xticks(range(1, K + 1)); ax.set_xlim(0.5, K + 0.5); ax.set_xlabel("rank"); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); strip(ax, left=False)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="D", color=ORANGE, ls="", ms=6, label="official rank (full cohort)"), Line2D([], [], marker="o", color="#fcfcfb", mec=BLUE, mew=1.6, ls="", ms=7, label="median bootstrap rank"), Line2D([], [], color=BLUE, lw=2, alpha=0.45, label="95% interval over 1,000 patient bootstraps")], loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Ranking stability: the complete scoring was repeated on 1,000 resamples of the 80 patients", loc="left", fontsize=9.5, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_rank_stability.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)

# 4. accuracy vs preservation: why the cap and gate exist ---------------------------------------
def desc(t):
    rows = [r for r in csv.DictReader(open(H / M[t]["results"])) if r["status"] == "ok"]
    return np.mean([float(r["CT_DSC_combined"]) for r in rows]), np.median([float(r["MTV_percent_error"]) for r in rows])
fig, ax = plt.subplots(figsize=(8.2, 5.0))
groups = {"participants": ([t for t in teams], BLUE), "organizer baselines": ([t for t in A["official"] if t.startswith("BASELINE")], ORANGE), "reference transforms": (["REF_AFFINE_ONLY", "REF_IDENTITY"], AQUA)}
OFFSETS = {"BASELINE_TM_IO": (4, -9), "BASELINE_TM_IO_SVF": (4, 4), "BASELINE_TM": (-6, 2), "housheng": (5, -3), "abakei": (-6, 2), "tinymilky": (5, -2), "koegl": (5, 5), "mbeguin": (5, -9), "jbgeb": (-6, 2), "bailiangj": (5, 4), "BASELINE_VXM_IO": (-6, -4), "adidukre": (5, -8), "neeldey": (5, -4), "longlai0000": (5, -8), "lukasf98": (-6, 2)}
names = {"BASELINE_CONVEXADAM": "ConvexAdam", "BASELINE_TM_IO_SVF": "TransMorph+IO+SVF", "BASELINE_TM_IO": "TransMorph+IO", "BASELINE_TM": "TransMorph", "BASELINE_VXM_IO": "VoxelMorph+IO", "BASELINE_VXM": "VoxelMorph", "REF_AFFINE_ONLY": "affine only", "REF_IDENTITY": "identity", "EXP130_v2": "MIAgent"}
for gname, (members, col) in groups.items():
    pts = [desc(t) for t in members]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46, color=col, edgecolor="#fcfcfb", linewidth=1, label=gname, zorder=3)
    for t, (x, v) in zip(members, pts):
        dx, dy = OFFSETS.get(t, (4, 3))
        ax.annotate(names.get(t, t), (x, v), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha="right" if dx < 0 else "left")
ax.set_xlabel("CT organ Dice (cohort mean) — higher is better"); ax.set_ylabel("|ΔMTV| % (cohort median) — lower is better")
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); strip(ax)
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
ax.set_title("Lesion preservation is trivially perfect without deformation: identity and affine-only sit at ~0 % MTV change.\nHence the cap (preservation ≤ accuracy) and the gate (must beat ConvexAdam on ≥ 2 accuracy metrics).", loc="left", fontsize=9, color=INK2)
fig.tight_layout(); fig.savefig(OUT / "fig_accuracy_vs_preservation.png", dpi=200, facecolor="#fcfcfb"); plt.close(fig)
print("figures written to", OUT)
