"""
plot_phase2_results.py — Phase 2 Training & Evaluation Visualisations
======================================================================
Generates five publication-quality figures from real Phase 2 run data:

  Fig 1  — Training loss curve (20 epochs)
  Fig 2  — Training accuracy curve
  Fig 3  — Zero-Day Tier 2 cosine similarity & escalation rates
  Fig 4  — Ingestion c_NLP confidence distribution (Phase 1 data)
  Fig 5  — threshold_weight breakdown (ACCEPTED-full / partial / FLAGGED)

Usage:
    python -X utf8 plot_phase2_results.py
Outputs saved to: plots/
"""

import os
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic run data (loads from actual training/eval runs)
# ─────────────────────────────────────────────────────────────────────────────

# Defaults (from screenshot / earlier run)
EPOCHS = list(range(1, 21))
TRAIN_LOSS = [
    0.6005, 0.2746, 0.1794, 0.1460, 0.1287,
    0.1152, 0.1088, 0.1003, 0.0918, 0.0886,
    0.0859, 0.0794, 0.0741, 0.0714, 0.0696,
    0.0643, 0.0632, 0.0617, 0.0642, 0.0646,
]
TRAIN_ACC = [
    0.6723, 0.8806, 0.9301, 0.9456, 0.9527,
    0.9591, 0.9626, 0.9662, 0.9696, 0.9710,
    0.9721, 0.9752, 0.9773, 0.9782, 0.9798,
    0.9809, 0.9810, 0.9815, 0.9815, 0.9813,
]
VAL_EPOCHS     = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
VAL_LOSS       = [2.8322, 5.4113, 7.3838, 8.3979, 9.5185,
                  10.2245, 10.9594, 11.7155, 11.8160, 12.0641]
VAL_HITS_AT_10 = [1.0] * 10
LR_SCHEDULE = [
    3.45e-4, 6.90e-4, 1.00e-3, 9.90e-4, 9.63e-4,
    9.21e-4, 8.65e-4, 7.96e-4, 7.17e-4, 6.32e-4,
    5.41e-4, 4.50e-4, 3.59e-4, 2.74e-4, 1.97e-4,
    1.29e-4, 7.40e-5, 3.34e-5, 8.41e-6, 0.00e+0,
]

# Locked seeded-run defaults (zero_day_test.py with per-function seeds 42/43/44)
ZD_COS_SIM     = [0.766, 0.483, 0.678]
ZD_ESC_RATE    = [0.192, 0.460, 0.000]
ZD_LINK_SCORE  = [0.8357, 0.7218, 0.0019]
TIER1_HITS_TEXT = "Hits@10 = 0.9900\n[PASS]  (target >= 0.70)"

# Track best epoch for dynamic annotation
BEST_EPOCH = 2  # default

# Try to load actual training metrics
try:
    with open("models/training_metrics.json", "r") as f:
        tm = json.load(f)
    EPOCHS = tm["epochs"]
    TRAIN_LOSS = tm["train_loss"]
    # Already raw fraction [0,1] from train_tgn.py — convert once to %
    TRAIN_ACC = [x * 100.0 for x in tm["train_acc"]]
    LR_SCHEDULE = tm["lr"]
    VAL_EPOCHS = tm["val_epochs"]
    VAL_LOSS = tm["val_loss"]
    VAL_HITS_AT_10 = tm["val_hits_at_10"]
    print("Loaded dynamic training metrics.")
except Exception:
    print("models/training_metrics.json not found. Using defaults.")
    TRAIN_ACC = [x * 100 for x in TRAIN_ACC]  # Ensure it's in % for legacy data

# Determine best checkpoint epoch from val_loss (lowest)
if VAL_EPOCHS and VAL_LOSS:
    best_idx  = int(VAL_LOSS.index(min(VAL_LOSS)))
    BEST_EPOCH = VAL_EPOCHS[best_idx]
    BEST_VAL_LOSS = VAL_LOSS[best_idx]
else:
    BEST_EPOCH    = EPOCHS[0] if EPOCHS else 2
    BEST_VAL_LOSS = VAL_LOSS[0] if VAL_LOSS else 0.0

# Try to load checkpoint for best epoch confirmation
try:
    import torch as _torch
    _ckpt = _torch.load("models/tgn_phase2.pt", map_location="cpu", weights_only=False)
    BEST_EPOCH = _ckpt.get("epoch", BEST_EPOCH)
except Exception:
    pass  # Use val_loss-derived best epoch

# Try to load actual zero-day metrics
try:
    with open("models/zero_day_metrics.json", "r") as f:
        zdm = json.load(f)
    t2 = zdm["tier2"]
    
    def get_t2_metric(level_key, metric_key, default):
        for k, v in t2.items():
            if level_key in k:
                return v.get(metric_key, default)
        return default

    ZD_COS_SIM = [
        get_t2_metric("level1", "mean_cosine_sim", ZD_COS_SIM[0]),
        get_t2_metric("level2", "mean_cosine_sim", ZD_COS_SIM[1]),
        get_t2_metric("level3", "mean_cosine_sim", ZD_COS_SIM[2])
    ]
    ZD_ESC_RATE = [
        get_t2_metric("level1", "escalation_rate", ZD_ESC_RATE[0]),
        get_t2_metric("level2", "escalation_rate", ZD_ESC_RATE[1]),
        get_t2_metric("level3", "escalation_rate", ZD_ESC_RATE[2])
    ]
    ZD_LINK_SCORE = [
        get_t2_metric("level1", "mean_link_score", ZD_LINK_SCORE[0]),
        get_t2_metric("level2", "mean_link_score", ZD_LINK_SCORE[1]),
        get_t2_metric("level3", "mean_link_score", ZD_LINK_SCORE[2])
    ]
    
    t1 = zdm["tier1"]
    if t1 and t1.get("hits_at_k") is not None:
        passed_str = "PASS" if t1.get("passed", False) else "FAIL"
        TIER1_HITS_TEXT = f"Hits@10 = {t1['hits_at_k']:.4f}\n[{passed_str}]  (target >= 0.70)"
    
    print("Loaded dynamic zero-day metrics.")
except Exception as e:
    print(f"models/zero_day_metrics.json not found or error ({e}). Using defaults.")

# Threshold weight breakdown (from training output)
THRESH_LABELS  = ["ACCEPTED-full\n(w=1.0)", "ACCEPTED-partial\n(w=c_nlp)", "FLAGGED\n(w=0.3)"]
THRESH_COUNTS  = [35640, 5281, 725]
THRESH_COLORS  = ["#2ecc71", "#f39c12", "#e74c3c"]

# Ingestion c_NLP distribution (from Phase 1 summary)
CNLP_LABELS  = ["<0.50\nREJECTED", "0.50–0.70\nFLAGGED", "0.70–0.90\nACCEPT-partial", "≥0.90\nACCEPT-full"]
CNLP_COUNTS  = [616, 1541, 9768, 25673]
CNLP_COLORS  = ["#c0392b", "#e67e22", "#3498db", "#27ae60"]
CNLP_TOTAL   = sum(CNLP_COUNTS)

# Zero-Day Tier 2 results labels
ZD_LEVELS      = ["L1: MITRE\nATT&CK Chains", "L2: CyberBattle\nSim", "L3: Adversarial\n(FLAGGED)"]

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0f1117"
PANEL_BG   = "#1a1d27"
ACCENT1    = "#7c6af7"   # purple
ACCENT2    = "#38c9b0"   # teal
ACCENT3    = "#f0a500"   # amber
ACCENT4    = "#e05252"   # red
GRID_COLOR = "#2a2d3a"
TEXT_COLOR = "#e8eaf0"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "axes.titlecolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_COLOR,
    "lines.linewidth":   2.2,
})

OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 + 2: Training curves (loss, accuracy, LR) — combined 2-panel
# ─────────────────────────────────────────────────────────────────────────────

# ── Dynamic axis limits ───────────────────────────────────────────────────────
MAX_EPOCH   = max(EPOCHS) if EPOCHS else 20
FINAL_ACC   = TRAIN_ACC[-1] if TRAIN_ACC else 98.0   # already in %
FINAL_LOSS  = TRAIN_LOSS[-1] if TRAIN_LOSS else 0.065

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    f"Phase 2A — ConfidenceWeightedTGN: Trained to convergence ({MAX_EPOCH} epochs); "
    f"best-generalising checkpoint (epoch {BEST_EPOCH}) selected via validation loss",
    fontsize=12, color=TEXT_COLOR, y=1.01
)

# — Panel A: Loss —
ax = axes[0]
ax.plot(EPOCHS, TRAIN_LOSS, color=ACCENT1, label="Train Loss", zorder=3)
ax.fill_between(EPOCHS, TRAIN_LOSS, alpha=0.15, color=ACCENT1)
ax.scatter(VAL_EPOCHS, VAL_LOSS, color=ACCENT3, s=55, zorder=5,
           label="Val Loss (BCE)", marker="D")
ax.plot(VAL_EPOCHS, VAL_LOSS, color=ACCENT3, linestyle="--", alpha=0.7, zorder=4)
ax.axhline(y=FINAL_LOSS, color=ACCENT2, linestyle=":", alpha=0.5, linewidth=1.2)

# Dynamic annotation: final train loss position
_loss_text_x = max(1, MAX_EPOCH - 6)
_loss_text_y = max(VAL_LOSS) * 0.55 if VAL_LOSS else 2.5
ax.annotate(
    f"Final train loss\n{FINAL_LOSS:.4f}",
    xy=(MAX_EPOCH, FINAL_LOSS),
    xytext=(_loss_text_x, _loss_text_y),
    color=ACCENT2, fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.2)
)
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE Loss")
ax.set_title("Loss Curves — Train drops; Val diverges (expected overfitting pattern)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.4)
ax.set_xlim(1, MAX_EPOCH)   # ← dynamic, not hardcoded 20

# Dynamic best-checkpoint annotation
if VAL_LOSS and BEST_EPOCH in VAL_EPOCHS:
    _bidx     = VAL_EPOCHS.index(BEST_EPOCH)
    _best_vl  = VAL_LOSS[_bidx]
    _ann_x    = min(BEST_EPOCH + 1, MAX_EPOCH)
    _ann_y    = _best_vl + max(VAL_LOSS) * 0.08
    ax.annotate(
        f"Best ckpt\n(epoch {BEST_EPOCH})",
        xy=(BEST_EPOCH, _best_vl),
        xytext=(_ann_x, _ann_y),
        color=ACCENT3, fontsize=8.5,
        arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=1.0)
    )

# — Panel B: Accuracy + LR —
# TRAIN_ACC is ALREADY in % — do NOT multiply by 100 again here
ax2 = axes[1]
ax2.plot(EPOCHS, TRAIN_ACC, color=ACCENT2, label="Train Acc (%)", zorder=3)
ax2.fill_between(EPOCHS, TRAIN_ACC, alpha=0.15, color=ACCENT2)

ax3 = ax2.twinx()
ax3.tick_params(colors=TEXT_COLOR)
ax3.spines["right"].set_color(GRID_COLOR)
ax3.plot(EPOCHS, [lr * 1e3 for lr in LR_SCHEDULE], color=ACCENT3,
         linestyle="--", linewidth=1.5, label="LR (×10⁻³)", alpha=0.8)
ax3.set_ylabel("Learning Rate (×10⁻³)", color=ACCENT3)
ax3.tick_params(axis="y", labelcolor=ACCENT3)

_acc_ref = min(98.0, FINAL_ACC)   # reference line at min(98%, final)
ax2.axhline(y=_acc_ref, color="#ffffff", linestyle=":", linewidth=0.8, alpha=0.4)
_ann_acc_x = max(1, MAX_EPOCH - 7)
ax2.annotate(
    f"{FINAL_ACC:.1f}% final",
    xy=(MAX_EPOCH, FINAL_ACC),
    xytext=(_ann_acc_x, max(60, FINAL_ACC - 8)),
    color=TEXT_COLOR, fontsize=8.5,
    arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.0)
)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training Accuracy & LR Schedule (checkpoint = epoch 2, lowest val loss)")
ax2.set_xlim(1, MAX_EPOCH)   # ← dynamic, not hardcoded 20
ax2.set_ylim(60, 102)
ax2.grid(True, alpha=0.4)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

plt.tight_layout()
out1 = OUT_DIR / "fig1_training_curves.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {out1}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Zero-Day Tier 2 — Cosine Similarity + Escalation Rate + Link Score
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle(
    "Phase 2B — Hybrid Zero-Day Evaluation (Tier 2: Structure-Novelty Spectrum) "
    "[Seeded: reproducible across re-runs]",
    fontsize=12, color=TEXT_COLOR, y=1.01
)

x = np.arange(len(ZD_LEVELS))
BAR_W = 0.55
LEVEL_COLORS = [ACCENT2, ACCENT3, ACCENT4]

# Cosine Similarity
ax = axes[0]
bars = ax.bar(x, ZD_COS_SIM, width=BAR_W, color=LEVEL_COLORS, alpha=0.88, zorder=3)
ax.axhline(y=0.40, color="#ffffff", linestyle="--", linewidth=1.5, alpha=0.7, label="OOD threshold (0.40)")
for bar, val in zip(bars, ZD_COS_SIM):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(ZD_LEVELS, fontsize=9)
ax.set_ylabel("Mean Cosine Similarity")
ax.set_title("Cosine Similarity to\nMalicious Cluster Centroid")
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.4)

# Escalation Rate
ax = axes[1]
esc_pct = [e * 100 for e in ZD_ESC_RATE]
bars = ax.bar(x, esc_pct, width=BAR_W, color=LEVEL_COLORS, alpha=0.88, zorder=3)
ax.axhline(y=0, color=GRID_COLOR, linewidth=0.8)
for bar, val in zip(bars, esc_pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(ZD_LEVELS, fontsize=9)
ax.set_ylabel("Escalation Rate (%)")
ax.set_title("OOD Escalation Rate\n(Nodes Flagged for Human Review)")
ax.set_ylim(0, 70)
ax.grid(True, axis="y", alpha=0.4)

# Mean Link Score
ax = axes[2]
bars = ax.bar(x, ZD_LINK_SCORE, width=BAR_W, color=LEVEL_COLORS, alpha=0.88, zorder=3)
ax.axhline(y=0.5, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.5, label="Decision boundary (0.5)")
for bar, val in zip(bars, ZD_LINK_SCORE):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, color=TEXT_COLOR)
ax.set_xticks(x)
ax.set_xticklabels(ZD_LEVELS, fontsize=9)
ax.set_ylabel("Mean Link Score (sigmoid)")
ax.set_title("Mean Link Prediction Score\n(Confidence of edge being real CTI)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
out2 = OUT_DIR / "fig2_zero_day_tier2.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: c_NLP Confidence Distribution (Phase 1 TIRE ingestion)
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Phase 1 Data Quality — c_NLP Confidence Distribution (37,598 Triplets)",
             fontsize=13, color=TEXT_COLOR, y=1.01)

# Bar chart
bars = ax1.bar(CNLP_LABELS, CNLP_COUNTS, color=CNLP_COLORS, alpha=0.88, zorder=3, width=0.6)
for bar, cnt in zip(bars, CNLP_COUNTS):
    pct = cnt / CNLP_TOTAL * 100
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,
             f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
ax1.set_ylabel("Triplet Count")
ax1.set_title("Confidence Bucket Distribution")
ax1.set_ylim(0, max(CNLP_COUNTS) * 1.18)
ax1.grid(True, axis="y", alpha=0.4)

# Donut chart
wedges, texts, autotexts = ax2.pie(
    CNLP_COUNTS, labels=None,
    colors=CNLP_COLORS, autopct="%1.1f%%",
    startangle=140, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
)
for at in autotexts:
    at.set_color(TEXT_COLOR)
    at.set_fontsize(9.5)

ax2.legend(
    wedges, CNLP_LABELS,
    loc="lower center", ncol=2, fontsize=8.5,
    bbox_to_anchor=(0.5, -0.18),
)
ax2.set_title("Confidence Tier Breakdown")
# Centre annotation
ax2.text(0, 0, "37,598\ntriplets", ha="center", va="center",
         fontsize=11, color=TEXT_COLOR, fontweight="bold")

plt.tight_layout()
out3 = OUT_DIR / "fig3_cnlp_distribution.png"
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {out3}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: threshold_weight Breakdown (how edges enter TGN)
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Phase 2A — Edge threshold_weight Distribution (41,646 TKG Edges fed to TGN)",
             fontsize=13, color=TEXT_COLOR, y=1.01)

total_edges = sum(THRESH_COUNTS)

# Horizontal bar
ax1.barh(THRESH_LABELS, THRESH_COUNTS, color=THRESH_COLORS, alpha=0.88, zorder=3, height=0.5)
for i, (cnt, lbl) in enumerate(zip(THRESH_COUNTS, THRESH_LABELS)):
    pct = cnt / total_edges * 100
    ax1.text(cnt + 200, i, f"{cnt:,}  ({pct:.1f}%)",
             va="center", fontsize=10, color=TEXT_COLOR)
ax1.set_xlabel("Edge Count")
ax1.set_title("Edge Count by Weight Tier")
ax1.set_xlim(0, max(THRESH_COUNTS) * 1.25)
ax1.grid(True, axis="x", alpha=0.4)
ax1.invert_yaxis()

# Stacked horizontal for visual proportion
proportions = [c / total_edges for c in THRESH_COUNTS]
left = 0
for prop, color, label in zip(proportions, THRESH_COLORS, THRESH_LABELS):
    ax2.barh(["Edge\nComposition"], prop * 100, left=left * 100,
             color=color, alpha=0.88, height=0.4, label=label)
    if prop > 0.02:
        ax2.text(left * 100 + prop * 50, 0,
                 f"{prop * 100:.1f}%", ha="center", va="center",
                 fontsize=9.5, color=DARK_BG, fontweight="bold")
    left += prop

ax2.set_xlabel("Percentage of TKG Edges (%)")
ax2.set_title("Stacked Edge Composition")
ax2.set_xlim(0, 100)
ax2.legend(loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.28))
ax2.grid(True, axis="x", alpha=0.4)
ax2.set_yticks([])

# Key stats box
final_val_hits = VAL_HITS_AT_10[-1] if VAL_HITS_AT_10 else 1.0
textstr = (f"Total edges: {total_edges:,}\n"
           f"FLAGGED penalised (×0.3): {THRESH_COUNTS[2]:,}\n"
           f"Hits@10 across {MAX_EPOCH} epochs: {final_val_hits:.3f}")
props = dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG, edgecolor=ACCENT1, alpha=0.9)
ax2.text(0.97, 0.97, textstr, transform=ax2.transAxes, fontsize=8.5,
         verticalalignment="top", horizontalalignment="right", bbox=props, color=TEXT_COLOR)

plt.tight_layout()
out4 = OUT_DIR / "fig4_threshold_weights.png"
plt.savefig(out4, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {out4}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Phase 2B Tier 1 — Ransomware Label-Novelty Summary
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(DARK_BG)

categories = ["Ransomware\nNodes Masked", "Held-out\nEdges", "Edges\nEvaluated"]
values     = [56, 754, 200]
bar_colors = [ACCENT1, ACCENT3, ACCENT2]

bars = ax.bar(categories, values, color=bar_colors, alpha=0.88, width=0.45, zorder=3)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:,}", ha="center", va="bottom", fontsize=12,
            color=TEXT_COLOR, fontweight="bold")

# Hits@10 annotation
ax.annotate("",
            xy=(2.25, 200), xytext=(2.25, 600),
            arrowprops=dict(arrowstyle="<->", color=TEXT_COLOR, lw=1.2))

props = dict(boxstyle="round,pad=0.7", facecolor=ACCENT2, edgecolor=ACCENT2, alpha=0.15)
ax.text(0.72, 0.72, TIER1_HITS_TEXT, transform=ax.transAxes, fontsize=13,
        ha="center", va="center", color=ACCENT2, fontweight="bold", bbox=props)

ax.set_title(
    "Phase 2B Tier 1 — Ransomware Label-Novelty Test\n"
    "(Ransomware edges masked at inference — model never saw these edges during encoding)",
    fontsize=11
)
ax.set_ylabel("Count")
ax.set_ylim(0, 900)
ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
out5 = OUT_DIR / "fig5_tier1_ransomware.png"
plt.savefig(out5, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved: {out5}")

print(f"\n[SUCCESS] All plots saved to: {OUT_DIR.resolve()}/")
