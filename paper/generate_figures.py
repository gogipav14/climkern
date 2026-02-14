#!/usr/bin/env python3
"""
Generate publication-quality figures for the ClimKern-Retune paper.

Produces 6 figures illustrating the two-stage kernel harmonization pipeline,
Q² dashboard, per-regime performance, retune comparison, spread reduction,
and kernel weights.

Usage:
    python paper/generate_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from numpy.typing import NDArray

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from kernel_harmonizer import KernelRegimeHarmonizer, Q2Dashboard
from validate_real_data import _generate_multi_kernel_data
from compute_real_q2 import compute_all_kernel_predictions, compute_real_q2
from tunable_kernel import TunableKernel, KernelConfig
from nipals_pls import ConstrainedNipalsPLS

# ── Global style ──────────────────────────────────────────────
FIGDIR = Path(__file__).resolve().parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Colors
C_KERNEL = "#9e9e9e"       # gray — individual kernels
C_BASELINE = "#42a5f5"     # blue — simple mean
C_STAGE1_G = "#66bb6a"     # green — global PLS
C_STAGE1_B = "#81c784"     # light green — SIMCA blend
C_RETUNE_A = "#ef5350"     # red
C_RETUNE_B = "#ff7043"     # orange
C_RETUNE_C = "#ffa726"     # amber
C_RETUNE_D = "#ffca28"     # yellow
C_RETUNE = [C_RETUNE_A, C_RETUNE_B, C_RETUNE_C, C_RETUNE_D]

KERNEL_NAMES = [
    "BMRC", "CAM3", "CAM5", "CERES", "CloudSat",
    "ECHAM6", "ECMWF-RRTM", "ERA5", "GFDL", "HadGEM2", "HadGEM3-GA7.1",
]


# ── Data generation ──────────────────────────────────────────
def fit_pipeline_real():
    """Load real CERES+NCEP data, fit two-stage pipeline, return dashboard + extras."""
    data_path = _root / "data" / "merged_CERES_NCEP_2003-2020.nc"
    kernel_dir = _root / "climkern" / "data" / "data" / "kernels"

    if not data_path.exists() or not kernel_dir.exists():
        print("  Real data not available, falling back to synthetic data")
        return fit_pipeline_synthetic()

    data = compute_all_kernel_predictions(data_path, kernel_dir)
    X = data["X_total"]
    Y = data["Y"][:, 0:1]  # LW only
    n = len(X)
    n_train = int(0.8 * n)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    harmonizer = KernelRegimeHarmonizer(n_components=3, min_samples_per_regime=30)
    harmonizer.fit_all(X_train, Y_train, data["kernel_names"])
    dashboard = harmonizer.q2_dashboard(X_test, Y_test)

    return harmonizer, dashboard, X_train, X_test, Y_train, Y_test, data


def fit_pipeline_synthetic():
    """Generate synthetic data, fit two-stage pipeline, return dashboard + extras."""
    mk_data = _generate_multi_kernel_data(n_samples=2000, n_kernels=11)
    X = mk_data["X_total"]
    Y = mk_data["Y"][:, 0:1]  # LW only
    n = len(X)
    n_train = int(0.8 * n)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    harmonizer = KernelRegimeHarmonizer(n_components=3, min_samples_per_regime=10)
    harmonizer.fit_all(X_train, Y_train, KERNEL_NAMES)
    dashboard = harmonizer.q2_dashboard(X_test, Y_test)

    return harmonizer, dashboard, X_train, X_test, Y_train, Y_test, mk_data


# ── Step 2: Data-driven kernel fitting ───────────────────────
def fit_step2_real():
    """Load real CERES+NCEP data, fit Step 2 TunableKernel, return kernel + data."""
    from validate_real_data import load_real_data

    real_data = load_real_data(_root / "data")
    if real_data is None:
        print("  Step 2: Real data not available, skipping")
        return None

    X, Y = real_data["X"], real_data["Y"]
    feature_names = real_data["feature_names"]
    n = len(X)
    n_train = int(0.8 * n)

    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]

    # Mean-center
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_train_c = X_train - X_mean
    X_test_c = X_test - X_mean

    config = KernelConfig(
        n_components=10,
        use_surface_constraint=False,
        use_toa_constraint=False,
        use_conservation_constraint=False,
    )
    kernel = TunableKernel(config=config)
    kernel.fit(X_train_c, Y_train - Y_mean, feature_names=feature_names)

    return {
        "kernel": kernel,
        "X_train_c": X_train_c,
        "X_test_c": X_test_c,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "Y_mean": Y_mean,
        "X_mean": X_mean,
        "feature_names": feature_names,
    }


# ══════════════════════════════════════════════════════════════
# Figure 1: Two-Stage Pipeline Architecture
# ══════════════════════════════════════════════════════════════
def fig1_pipeline():
    """Architecture diagram of the two-stage ClimKern-Retune pipeline."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # ── Helper functions ──
    def box(x, y, w, h, text, color="#e3f2fd", ec="#1565c0", lw=1.2,
            fontsize=8, fontstyle="normal", fontweight="normal", alpha=1.0):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=ec, linewidth=lw, alpha=alpha,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                fontstyle=fontstyle, fontweight=fontweight,
                wrap=True)

    def arrow(x1, y1, x2, y2, color="#37474f"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=1.5,
                connectionstyle="arc3,rad=0",
            ),
        )

    def section_label(x, y, text, color="#37474f"):
        ax.text(x, y, text, fontsize=11, fontweight="bold", color=color,
                ha="left", va="center")

    # ── Title ──
    ax.text(5, 6.7, "ClimKern-Retune: Two-Stage Kernel Harmonization Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # ── Stage 1 header ──
    section_label(0.2, 6.2, "Stage 1: ClimKern Foundation", "#1565c0")
    ax.axhline(y=6.05, xmin=0.02, xmax=0.98, color="#1565c0", lw=0.5, ls="--")

    # Input
    box(0.3, 5.2, 2.0, 0.7, "11 Kernel\nPredictions\nX  (n x 11)",
        color="#e8f5e9", ec="#2e7d32", fontsize=8)

    # Global PLS
    arrow(2.3, 5.55, 3.1, 5.55)
    box(3.1, 5.2, 1.8, 0.7, "Global PLS\n(Constrained\nNIPALS)",
        color="#e3f2fd", ec="#1565c0", fontsize=8)

    # CERES target
    box(0.3, 4.2, 2.0, 0.6, "CERES Obs.\nY  (n x 2)",
        color="#fff3e0", ec="#e65100", fontsize=8)
    arrow(1.3, 4.8, 1.3, 5.2)

    # KernelRegimeClassifier
    arrow(4.0, 4.8, 4.0, 5.2)
    box(3.1, 4.1, 1.8, 0.7, "Kernel Regime\nClassifier\n(argmin residual)",
        color="#fce4ec", ec="#c62828", fontsize=7.5)
    arrow(2.3, 4.5, 3.1, 4.5)

    # SIMCA
    arrow(4.9, 4.45, 5.6, 4.45)
    box(5.6, 4.1, 1.7, 0.7, "SIMCA\n(11 PCA models,\nsoft proba.)",
        color="#f3e5f5", ec="#6a1b9a", fontsize=7.5)

    # Per-regime PLS
    arrow(7.3, 4.45, 7.8, 4.45)
    box(7.8, 4.1, 1.8, 0.7, "Per-Regime\nPLS (11 models,\nregime weights)",
        color="#e3f2fd", ec="#1565c0", fontsize=7.5)

    # Stage 1 outputs
    arrow(4.9, 5.55, 5.6, 5.55)
    box(5.6, 5.2, 1.5, 0.7, "Baseline Q\u00b2\n(Global PLS)",
        color="#e8eaf6", ec="#283593", fontsize=8, fontweight="bold")

    arrow(7.3, 5.55, 7.8, 5.55)
    box(7.8, 5.2, 1.8, 0.7, "Regime Q\u00b2\n(SIMCA blend)",
        color="#e8eaf6", ec="#283593", fontsize=8)

    # ── Stage 2 header ──
    section_label(0.2, 3.5, "Stage 2: Retune (4 Approaches)", "#c62828")
    ax.axhline(y=3.35, xmin=0.02, xmax=0.98, color="#c62828", lw=0.5, ls="--")

    # SIMCA structure feeds into Stage 2
    arrow(6.4, 4.1, 6.4, 3.1)
    ax.text(6.6, 3.6, "SIMCA\nstructure", fontsize=7, color="#6a1b9a",
            ha="left", va="center", fontstyle="italic")

    # Four approaches
    approaches = [
        ("A: SIMCA\nFeatures\n(X + proba\n\u2192 PLS)", "#ef5350", "#b71c1c"),
        ("B: Regime-\nWeighted\nPLS\n(sqrt(w)\u00b7X)", "#ff7043", "#bf360c"),
        ("C: Two-Pass\nIterative\n(PLS \u2194\nSIMCA)", "#ffa726", "#e65100"),
        ("D: Regime\nCollapse\n(\u03a3 n\u2096/N \u00b7 B\u2096)", "#ffca28", "#f57f17"),
    ]
    x_starts = [0.5, 2.8, 5.1, 7.4]
    for i, (text, fc, ec) in enumerate(approaches):
        box(x_starts[i], 1.9, 2.0, 1.1, text, color=fc, ec=ec,
            fontsize=7.5, alpha=0.85)

    # Arrows from SIMCA to each approach
    for x in x_starts:
        arrow(6.4, 3.1, x + 1.0, 3.0)

    # Comparison box
    arrow(1.5, 1.9, 4.0, 1.1)
    arrow(3.8, 1.9, 4.5, 1.1)
    arrow(6.1, 1.9, 5.5, 1.1)
    arrow(8.4, 1.9, 6.0, 1.1)

    box(3.5, 0.3, 3.0, 0.8, "Q\u00b2 Dashboard\n+ Thermodynamic Compliance\n\u2192 Best Method",
        color="#e8f5e9", ec="#1b5e20", fontsize=8.5, fontweight="bold")

    fig.savefig(FIGDIR / "fig1_pipeline.pdf")
    plt.close(fig)
    print("  fig1_pipeline.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 2: Q² Dashboard — All Methods Compared
# ══════════════════════════════════════════════════════════════
def fig2_q2_dashboard(dashboard: Q2Dashboard):
    """Horizontal bar chart comparing Q² across all methods."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Collect data in display order (bottom to top for barh)
    labels, values, colors = [], [], []

    # Individual kernels (sorted ascending by Q²)
    sorted_kernels = sorted(dashboard.q2_per_kernel.items(), key=lambda x: x[1])
    for name, q2 in sorted_kernels:
        labels.append(name)
        values.append(q2)
        colors.append(C_KERNEL)

    # Separator
    labels.append("")
    values.append(0)
    colors.append("white")

    # Baselines
    labels.append("Simple Mean")
    values.append(dashboard.q2_simple_mean)
    colors.append(C_BASELINE)

    labels.append("")
    values.append(0)
    colors.append("white")

    # Stage 1
    labels.append("Global PLS (Stage 1)")
    values.append(dashboard.q2_global_pls)
    colors.append(C_STAGE1_G)

    labels.append("SIMCA Blend (Stage 1)")
    values.append(dashboard.q2_regime_blend)
    colors.append(C_STAGE1_B)

    labels.append("")
    values.append(0)
    colors.append("white")

    # Stage 2 retune
    retune_data = [
        ("A: SIMCA Features", dashboard.q2_retune_A, C_RETUNE_A),
        ("B: Regime-Weighted", dashboard.q2_retune_B, C_RETUNE_B),
        ("C: Two-Pass Iter.", dashboard.q2_retune_C, C_RETUNE_C),
        ("D: Regime Collapse", dashboard.q2_retune_D, C_RETUNE_D),
    ]
    for name, q2, c in retune_data:
        labels.append(name)
        values.append(q2)
        colors.append(c)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Value labels
    for i, (v, lbl) in enumerate(zip(values, labels)):
        if lbl == "":
            continue
        offset = 0.02 if v >= 0 else -0.02
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:.3f}", va="center", ha=ha, fontsize=8, color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("$Q^2$ (Predictive $R^2$)")
    ax.set_title("$Q^2$ Dashboard: All Steps and Versions (Real Data)")
    vmin = min(v for v, l in zip(values, labels) if l != "")
    vmax = max(v for v, l in zip(values, labels) if l != "")
    margin = (vmax - vmin) * 0.1
    ax.set_xlim(vmin - margin, vmax + margin)
    ax.axvline(0, color="#999", ls=":", lw=0.8)

    # Section labels
    n_kernels = len(dashboard.q2_per_kernel)
    ax.text(0.02, n_kernels // 2, "Individual\nKernels",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="#666", fontstyle="italic", va="center")

    # Best marker
    best_idx = [i for i, l in enumerate(labels) if l != "" and values[i] == max(
        v for v, l2 in zip(values, labels) if l2 != "")]
    if best_idx:
        ax.plot(values[best_idx[0]], best_idx[0], "*", markersize=14,
                color="#ffd600", markeredgecolor="#333", markeredgewidth=0.5,
                zorder=5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_q2_dashboard.pdf")
    plt.close(fig)
    print("  fig2_q2_dashboard.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 3: Per-Regime Q² Variation
# ══════════════════════════════════════════════════════════════
def fig3_regime_q2(dashboard: Q2Dashboard):
    """Bar chart of Q² per SIMCA kernel regime."""
    fig, ax = plt.subplots(figsize=(7, 4))

    regimes = sorted(dashboard.q2_per_regime.keys())
    q2_vals = [dashboard.q2_per_regime[r] for r in regimes]
    regime_labels = [f"Regime {r}" for r in regimes]

    # Color by Q² magnitude (viridis colormap)
    norm = plt.Normalize(vmin=min(q2_vals) - 0.1, vmax=max(q2_vals) + 0.05)
    cmap = plt.cm.RdYlGn
    bar_colors = [cmap(norm(v)) for v in q2_vals]

    bars = ax.bar(range(len(regimes)), q2_vals, color=bar_colors,
                  edgecolor="white", linewidth=0.8, width=0.7)

    # Value labels on bars
    for i, v in enumerate(q2_vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Reference lines
    ax.axhline(dashboard.q2_global_pls, color=C_STAGE1_G, ls="--", lw=1.5,
               label=f"Global PLS ($Q^2$={dashboard.q2_global_pls:.3f})")
    ax.axhline(dashboard.q2_simple_mean, color=C_BASELINE, ls=":", lw=1.5,
               label=f"Simple Mean ($Q^2$={dashboard.q2_simple_mean:.3f})")

    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regime_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("$Q^2$")
    ax.set_title("Per-Regime $Q^2$ (SIMCA Kernel Regimes)")
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ymin = min(min(q2_vals), dashboard.q2_simple_mean) - 0.15
    ax.set_ylim(bottom=max(ymin, -2.0), top=max(q2_vals) + 0.15)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_regime_q2.pdf")
    plt.close(fig)
    print("  fig3_regime_q2.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 4: Retune Approach Multi-Metric Comparison
# ══════════════════════════════════════════════════════════════
def fig4_retune_comparison(dashboard: Q2Dashboard):
    """Grouped bar chart: 5 methods x 4 metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5))

    methods = ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]
    method_labels = ["Global\nPLS", "A: SIMCA\nFeatures", "B: Regime\nWeighted",
                     "C: Two-Pass\nIter.", "D: Regime\nCollapse"]
    method_colors = [C_STAGE1_G, C_RETUNE_A, C_RETUNE_B, C_RETUNE_C, C_RETUNE_D]

    # Metric 1: Q²
    q2_vals = [
        dashboard.q2_global_pls,
        dashboard.q2_retune_A,
        dashboard.q2_retune_B,
        dashboard.q2_retune_C,
        dashboard.q2_retune_D,
    ]
    axes[0].bar(range(5), q2_vals, color=method_colors, edgecolor="white", width=0.6)
    axes[0].set_title("$Q^2$", fontsize=10)
    q2_min = min(q2_vals)
    q2_max = max(q2_vals)
    q2_range = q2_max - q2_min if q2_max != q2_min else 0.1
    axes[0].set_ylim(q2_min - q2_range * 0.15, q2_max + q2_range * 0.15)
    axes[0].axhline(0, color="#999", ls=":", lw=0.8)
    for i, v in enumerate(q2_vals):
        offset = q2_range * 0.03
        axes[0].text(i, v + offset, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Metric 2: Stefan-Boltzmann residual (lower = better)
    sb_vals = [dashboard.sb_residual.get(m, 0) for m in methods]
    axes[1].bar(range(5), sb_vals, color=method_colors, edgecolor="white", width=0.6)
    axes[1].set_title("S-B Residual\n(lower = better)", fontsize=10)
    for i, v in enumerate(sb_vals):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Metric 3: Energy conservation residual (lower = better)
    ec_vals = [dashboard.energy_residual.get(m, 0) for m in methods]
    axes[2].bar(range(5), ec_vals, color=method_colors, edgecolor="white", width=0.6)
    axes[2].set_title("Energy Cons.\nResidual", fontsize=10)
    for i, v in enumerate(ec_vals):
        axes[2].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # Metric 4: RMSE reduction %
    red_vals = [dashboard.spread_reduction_pct.get(m, 0) for m in methods]
    axes[3].bar(range(5), red_vals, color=method_colors, edgecolor="white", width=0.6)
    axes[3].set_title("RMSE\nReduction %", fontsize=10)
    for i, v in enumerate(red_vals):
        axes[3].text(i, v + 0.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7)

    for ax in axes:
        ax.set_xticks(range(5))
        ax.set_xticklabels(method_labels, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Retune Approach Comparison: 4 Metrics", fontsize=12)
    fig.subplots_adjust(top=0.82, bottom=0.18, wspace=0.15)
    fig.savefig(FIGDIR / "fig4_retune_comparison.pdf")
    plt.close(fig)
    print("  fig4_retune_comparison.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 5: Spread Reduction — Before vs After
# ══════════════════════════════════════════════════════════════
def fig5_spread_reduction(dashboard: Q2Dashboard, harmonizer, X_test, Y_test):
    """Visualize interkernel spread reduction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # ── Left panel: Before vs After RMSE ──
    methods = ["global_pls", "retune_A", "retune_B", "retune_C", "retune_D"]
    method_labels = ["Global\nPLS", "Retune\nA", "Retune\nB", "Retune\nC", "Retune\nD"]
    method_colors = [C_STAGE1_G, C_RETUNE_A, C_RETUNE_B, C_RETUNE_C, C_RETUNE_D]

    rmse_before = dashboard.rmse_before
    rmse_after_vals = [dashboard.rmse_after.get(m, 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, [rmse_before] * len(methods),
                    width, label="Before (indiv. kernels)", color="#bdbdbd",
                    edgecolor="white")
    bars2 = ax1.bar(x + width / 2, rmse_after_vals,
                    width, label="After (harmonized)", color=method_colors,
                    edgecolor="white")

    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, fontsize=8)
    ax1.set_ylabel("RMSE (W/m$^2$)")
    ax1.set_title("RMSE: Before vs After Harmonization")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for i, v in enumerate(rmse_after_vals):
        pct = dashboard.spread_reduction_pct.get(methods[i], 0)
        ax1.text(i + width / 2, v + 0.02, f"-{pct:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="#c62828",
                 fontweight="bold")

    # ── Right panel: Residual distributions ──
    # Individual kernel residuals vs harmonized
    Y_flat = Y_test.ravel()

    # Individual kernel residuals (all 11)
    all_kernel_resid = []
    for k in range(X_test.shape[1]):
        resid = X_test[:, k] - Y_flat
        all_kernel_resid.extend(resid.tolist())

    # Harmonized residual (best method)
    Y_pred = harmonizer.predict(X_test, method="best")
    harm_resid = (Y_pred.ravel() - Y_flat).tolist()

    parts = ax2.violinplot(
        [all_kernel_resid, harm_resid],
        positions=[1, 2], showmeans=True, showmedians=True,
    )
    for pc in parts["bodies"]:
        pc.set_alpha(0.7)
    parts["bodies"][0].set_facecolor(C_KERNEL)
    parts["bodies"][1].set_facecolor(C_STAGE1_G)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Individual\nKernels (all 11)", "Harmonized\n(best method)"],
                        fontsize=9)
    ax2.set_ylabel("Residual (W/m$^2$)")
    ax2.set_title("Residual Distribution")
    ax2.axhline(0, color="#999", ls=":", lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_spread_reduction.pdf")
    plt.close(fig)
    print("  fig5_spread_reduction.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 6: VIP Scores + Kernel Weights
# ══════════════════════════════════════════════════════════════
def _compute_vip(pls_results) -> NDArray:
    """
    Compute Variable Importance in Projection (VIP) scores.

    VIP_j = sqrt(p * sum_a(SS_a * w_ja^2 / ||w_a||^2) / sum_a(SS_a))

    where SS_a = ||t_a||^2 * ||q_a||^2 is the sum of squares explained
    by component a, and w_ja is the weight of variable j in component a.
    """
    W = pls_results.x_weights      # (p, A) — PLS x-weights
    T = pls_results.x_scores       # (n, A) — PLS x-scores
    Q = pls_results.y_loadings     # (q, A) — PLS y-loadings
    p = W.shape[0]  # number of predictor variables
    A = W.shape[1]  # number of components

    # Sum of squares explained by each component
    SS = np.zeros(A)
    for a in range(A):
        SS[a] = np.dot(T[:, a], T[:, a]) * np.dot(Q[:, a], Q[:, a])

    SS_total = np.sum(SS)
    if SS_total == 0:
        return np.ones(p)

    # VIP for each variable
    vip = np.zeros(p)
    for j in range(p):
        s = 0.0
        for a in range(A):
            w_norm_sq = np.dot(W[:, a], W[:, a])
            if w_norm_sq > 0:
                s += SS[a] * (W[j, a] ** 2) / w_norm_sq
        vip[j] = np.sqrt(p * s / SS_total)

    return vip


def fig6_kernel_weights(harmonizer):
    """Two-panel figure: VIP scores (left) and regression coefficients (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Get PLS internals
    pls = harmonizer.global_pls_
    r = pls.results_
    vip = _compute_vip(r)

    weights = harmonizer.get_kernel_weights()  # (n_kernels, n_targets)
    w_lw = weights[:, 0] if weights.ndim > 1 else weights

    n_kernels = len(w_lw)
    names = KERNEL_NAMES[:n_kernels]

    # ── Left panel: VIP scores (sorted descending) ──
    order_vip = np.argsort(vip)[::-1]
    sorted_names_vip = [names[i] for i in order_vip]
    sorted_vip = vip[order_vip]

    # Color by importance: VIP > 1 = important (green), < 1 = less important (gray)
    vip_colors = [C_STAGE1_G if v >= 1.0 else C_KERNEL for v in sorted_vip]

    ax1.barh(range(n_kernels), sorted_vip[::-1], color=vip_colors[::-1],
             edgecolor="white", linewidth=0.5, height=0.6)

    for i, v in enumerate(sorted_vip[::-1]):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", ha="left", fontsize=8)

    ax1.set_yticks(range(n_kernels))
    ax1.set_yticklabels(sorted_names_vip[::-1], fontsize=9)
    ax1.set_xlabel("VIP Score")
    ax1.set_title("Variable Importance in Projection")
    ax1.axvline(1.0, color="#c62828", ls="--", lw=1.2, label="VIP = 1 threshold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Right panel: Regression coefficients ──
    order_w = np.argsort(w_lw)
    sorted_names_w = [names[i] for i in order_w]
    sorted_weights = w_lw[order_w]

    bar_colors = [C_STAGE1_G if w >= 0 else C_RETUNE_A for w in sorted_weights]

    ax2.barh(range(n_kernels), sorted_weights, color=bar_colors,
             edgecolor="white", linewidth=0.5, height=0.6)

    for i, w in enumerate(sorted_weights):
        offset = 0.003 if w >= 0 else -0.003
        ha = "left" if w >= 0 else "right"
        ax2.text(w + offset, i, f"{w:+.3f}", va="center", ha=ha, fontsize=8)

    ax2.set_yticks(range(n_kernels))
    ax2.set_yticklabels(sorted_names_w, fontsize=9)
    ax2.set_xlabel("PLS Regression Coefficient")
    ax2.set_title("Kernel Weights (LW Component)")
    ax2.axvline(0, color="#333", lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_kernel_weights.pdf")
    plt.close(fig)
    print("  fig6_kernel_weights.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 7½: Component Sweep (R² and Q² vs number of components)
# ══════════════════════════════════════════════════════════════
def fig7b_component_sweep(X_train, X_test, Y_train, Y_test):
    """Line plot of train R² and test Q² as function of PLS components."""
    from nipals_pls import ConstrainedNipalsPLS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    max_comp = min(8, X_train.shape[1])
    comps = list(range(1, max_comp + 1))
    r2_train, q2_test, delta_q2 = [], [], []

    for nc in comps:
        pls = ConstrainedNipalsPLS(n_components=nc)
        pls.fit(X_train, Y_train)

        Y_pred_tr = pls.predict(X_train)
        ss_res_tr = np.sum((Y_train - Y_pred_tr) ** 2)
        ss_tot_tr = np.sum((Y_train - np.mean(Y_train, axis=0)) ** 2)
        r2 = 1 - ss_res_tr / ss_tot_tr

        Y_pred_te = pls.predict(X_test)
        ss_res_te = np.sum((Y_test - Y_pred_te) ** 2)
        ss_tot_te = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
        q2 = 1 - ss_res_te / ss_tot_te

        r2_train.append(r2)
        q2_test.append(q2)
        delta_q2.append(q2 - q2_test[-2] if len(q2_test) > 1 else 0)

    # ── Left: R² and Q² vs components ──
    ax1.plot(comps, r2_train, "s-", color=C_RETUNE_B, markersize=6,
             label="Train $R^2$", linewidth=1.5)
    ax1.plot(comps, q2_test, "o-", color=C_STAGE1_G, markersize=6,
             label="Test $Q^2$", linewidth=1.5)
    ax1.axvline(3, color="#999", ls=":", lw=1, label="$A=3$ (used)")
    ax1.set_xlabel("Number of PLS Components ($A$)")
    ax1.set_ylabel("$R^2$ / $Q^2$")
    ax1.set_title("Model Performance vs Components")
    ax1.set_xticks(comps)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate key values
    for i, nc in enumerate(comps):
        if nc in (1, 3, 5, 8):
            ax1.annotate(f"{q2_test[i]:.3f}", xy=(nc, q2_test[i]),
                         xytext=(5, 8), textcoords="offset points",
                         fontsize=7, color=C_STAGE1_G)

    # ── Right: Marginal Q² gain per component ──
    ax2.bar(comps[1:], [q2_test[i] - q2_test[i - 1] for i in range(1, len(comps))],
            color=[C_STAGE1_G if i < 3 else C_KERNEL for i in range(len(comps) - 1)],
            edgecolor="white", width=0.6)
    ax2.axhline(0.001, color="#c62828", ls="--", lw=1, label="$\\Delta Q^2 = 0.001$")
    ax2.set_xlabel("Component Added")
    ax2.set_ylabel("$\\Delta Q^2$")
    ax2.set_title("Marginal $Q^2$ Gain per Component")
    ax2.set_xticks(comps[1:])
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i in range(1, len(comps)):
        dq = q2_test[i] - q2_test[i - 1]
        ax2.text(comps[i], dq + 0.0005, f"+{dq:.4f}", ha="center",
                 va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig8_component_sweep.pdf")
    plt.close(fig)
    print("  fig8_component_sweep.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 7: PLS Diagnostics (Scores, Hotelling T², DModX, Loadings)
# ══════════════════════════════════════════════════════════════
def _compute_hotellings_t2(T_scores: np.ndarray) -> np.ndarray:
    """Hotelling's T² for each observation: T² = Σ_a (t_ia / s_a)²."""
    s2 = np.var(T_scores, axis=0, ddof=1)
    s2[s2 == 0] = 1.0  # avoid division by zero
    return np.sum(T_scores**2 / s2, axis=1)


def _compute_dmodx(X: np.ndarray, T_scores: np.ndarray,
                   P_loadings: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
    """DModX: RMS of X-residuals after removing PLS components."""
    X_centered = X - x_mean
    X_hat = T_scores @ P_loadings.T
    E = X_centered - X_hat
    # Normalized RMS residual per observation
    p = X.shape[1]
    A = T_scores.shape[1]
    dof = max(p - A, 1)
    return np.sqrt(np.sum(E**2, axis=1) / dof)


def fig7_diagnostics(harmonizer, X_test, Y_test):
    """2x2 diagnostic panel: score plot, Hotelling T², DModX, loading plot."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    pls = harmonizer.global_pls_
    r = pls.results_

    # Compute test-set scores
    T_test = pls.transform(X_test)

    # Regime assignments for coloring (argmin residual, same as KernelRegimeClassifier)
    n_kernels = X_test.shape[1]
    kernel_names = KERNEL_NAMES[:n_kernels]
    Y_flat = Y_test[:, 0:1] if Y_test.ndim > 1 else Y_test[:, None]
    regime_ids = np.argmin(np.abs(X_test - Y_flat), axis=1)

    # ── (a) Score plot: t1 vs t2 colored by kernel regime ──
    ax = axes[0, 0]
    # Use a qualitative colormap for 11 regimes
    cmap = plt.cm.tab10
    unique_regimes = np.unique(regime_ids)

    # Subsample for readability (plot max 5000 points)
    n_plot = min(5000, len(T_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(T_test), n_plot, replace=False)

    scatter = ax.scatter(
        T_test[idx, 0], T_test[idx, 1],
        c=regime_ids[idx], cmap="tab10", s=4, alpha=0.4,
        vmin=-0.5, vmax=max(unique_regimes) + 0.5,
    )

    ax.set_xlabel("$t_1$ (Score 1)")
    ax.set_ylabel("$t_2$ (Score 2)")
    ax.set_title("(a) Score Plot Colored by Kernel Regime")
    ax.axhline(0, color="#999", ls=":", lw=0.5)
    ax.axvline(0, color="#999", ls=":", lw=0.5)

    # Compact legend: show regime names
    from matplotlib.lines import Line2D
    legend_elements = []
    for rid in unique_regimes[:11]:
        name = kernel_names[rid] if rid < len(kernel_names) else f"R{rid}"
        color = cmap(rid / max(unique_regimes.max(), 1))
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=5,
                                       label=name))
    ax.legend(handles=legend_elements, fontsize=6, loc="upper right",
              ncol=2, handletextpad=0.1, columnspacing=0.5)

    # ── (b) Hotelling's T² ──
    ax = axes[0, 1]
    T_train = r.x_scores  # training scores
    T2_train = _compute_hotellings_t2(T_train)
    T2_test = _compute_hotellings_t2(T_test)

    # Critical limit (95%): for large n, T² ~ chi²(A) * A*(n²-1)/(n*(n-A))
    A = T_train.shape[1]
    from scipy import stats
    T2_crit_95 = stats.chi2.ppf(0.95, A) * A

    ax.hist(T2_test, bins=80, density=True, color=C_STAGE1_G, alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Test set")
    ax.axvline(T2_crit_95, color="#c62828", ls="--", lw=1.5,
               label=f"95% limit ({T2_crit_95:.1f})")
    pct_above = 100 * np.mean(T2_test > T2_crit_95)
    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("Density")
    ax.set_title(f"(b) Hotelling's $T^2$ ({pct_above:.1f}% above 95% limit)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.percentile(T2_test, 99.5))

    # ── (c) DModX ──
    ax = axes[1, 0]
    x_mean = r.x_mean if r.x_mean is not None else np.mean(X_test, axis=0)
    P = r.x_loadings
    dmodx_test = _compute_dmodx(X_test, T_test, P, x_mean)

    # Color by regime
    ax.scatter(np.arange(n_plot), dmodx_test[idx], c=regime_ids[idx],
               cmap="tab10", s=4, alpha=0.4,
               vmin=-0.5, vmax=max(unique_regimes) + 0.5)

    # DModX critical limit: mean + 2*std of training DModX
    dmodx_train = _compute_dmodx(
        harmonizer._X_train_cache, T_train, P, x_mean)
    dcrit = np.mean(dmodx_train) + 2 * np.std(dmodx_train)
    ax.axhline(dcrit, color="#c62828", ls="--", lw=1.2,
               label=f"D-crit ({dcrit:.1f})")
    pct_above_d = 100 * np.mean(dmodx_test > dcrit)
    ax.set_xlabel("Observation index (subsampled)")
    ax.set_ylabel("DModX (RMS X-residual)")
    ax.set_title(f"(c) DModX ({pct_above_d:.1f}% above D-crit)")
    ax.legend(fontsize=8)

    # ── (d) Loading plot: p1 vs p2 with kernel labels ──
    ax = axes[1, 1]
    P = r.x_loadings  # (p, A)

    for j in range(n_kernels):
        ax.annotate(
            kernel_names[j],
            xy=(P[j, 0], P[j, 1]),
            fontsize=7, ha="center", va="bottom",
        )
    ax.scatter(P[:n_kernels, 0], P[:n_kernels, 1], s=40, c=C_STAGE1_G,
               edgecolor="#333", linewidth=0.5, zorder=5)

    # Draw unit circle reference (scaled)
    theta = np.linspace(0, 2 * np.pi, 100)
    r_max = max(np.max(np.abs(P[:, 0])), np.max(np.abs(P[:, 1]))) * 1.1
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta),
            color="#ccc", ls=":", lw=0.8)

    ax.set_xlabel("$p_1$ (Loading 1)")
    ax.set_ylabel("$p_2$ (Loading 2)")
    ax.set_title("(d) Loading Plot (Kernel Positions)")
    ax.axhline(0, color="#999", ls=":", lw=0.5)
    ax.axvline(0, color="#999", ls=":", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_diagnostics.pdf")
    plt.close(fig)
    print("  fig7_diagnostics.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 9: Step 2 — VIP Scores for 27 Atmospheric Features
# ══════════════════════════════════════════════════════════════
def fig9_step2_vip(step2_data):
    """Two-panel: VIP scores (left) and regression coefficients (right) for 27 features."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7))

    kernel = step2_data["kernel"]
    r = kernel.pls_model_.results_
    feature_names = step2_data["feature_names"]
    vip = _compute_vip(r)

    # Full regression coefficients: W @ diag(B_inner) @ Q'
    W = r.x_weights
    B = r.regression_matrix
    Q = r.y_loadings
    coeffs = W @ B @ Q.T  # (n_features, n_targets)
    coeff_lw = coeffs[:, 0]  # LW component

    n_feat = len(feature_names)

    # ── Left panel: VIP scores (sorted descending) ──
    order_vip = np.argsort(vip)[::-1]
    sorted_names = [feature_names[i] for i in order_vip]
    sorted_vip = vip[order_vip]

    # Color: T features=blue, q features=teal, surface/cloud=orange
    def _feat_color(name):
        if name.startswith("q_"):
            return "#26a69a"  # teal for humidity
        elif name == "cloud_fraction":
            return "#ff7043"  # orange
        elif name == "T_surface":
            return "#ffa726"  # amber
        else:
            return "#42a5f5"  # blue for temperature levels

    vip_colors = [_feat_color(n) for n in sorted_names]

    ax1.barh(range(n_feat), sorted_vip[::-1], color=vip_colors[::-1],
             edgecolor="white", linewidth=0.5, height=0.7)

    for i, v in enumerate(sorted_vip[::-1]):
        ax1.text(v + 0.02, i, f"{v:.2f}", va="center", ha="left", fontsize=7)

    ax1.set_yticks(range(n_feat))
    ax1.set_yticklabels(sorted_names[::-1], fontsize=7.5)
    ax1.set_xlabel("VIP Score")
    ax1.set_title("(a) Variable Importance in Projection")
    ax1.axvline(1.0, color="#c62828", ls="--", lw=1.2, label="VIP = 1 threshold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Feature-type legend
    from matplotlib.lines import Line2D
    type_legend = [
        Line2D([0], [0], color="#42a5f5", lw=6, label="Temperature"),
        Line2D([0], [0], color="#26a69a", lw=6, label="Humidity"),
        Line2D([0], [0], color="#ffa726", lw=6, label="Surface T"),
        Line2D([0], [0], color="#ff7043", lw=6, label="Cloud frac."),
    ]
    ax1.legend(handles=type_legend + [Line2D([0], [0], color="#c62828", ls="--", label="VIP = 1")],
               fontsize=7, loc="lower right")

    # ── Right panel: Regression coefficients (LW) ──
    order_w = np.argsort(coeff_lw)
    sorted_names_w = [feature_names[i] for i in order_w]
    sorted_coeff = coeff_lw[order_w]

    bar_colors_w = [_feat_color(n) for n in sorted_names_w]

    ax2.barh(range(n_feat), sorted_coeff, color=bar_colors_w,
             edgecolor="white", linewidth=0.5, height=0.7)

    for i, w in enumerate(sorted_coeff):
        offset = 0.001 if w >= 0 else -0.001
        ha = "left" if w >= 0 else "right"
        ax2.text(w + offset, i, f"{w:+.3f}", va="center", ha=ha, fontsize=7)

    ax2.set_yticks(range(n_feat))
    ax2.set_yticklabels(sorted_names_w, fontsize=7.5)
    ax2.set_xlabel("PLS Regression Coefficient (LW)")
    ax2.set_title("(b) Feature Contributions to $\\Delta R_{LW}$")
    ax2.axvline(0, color="#333", lw=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Step 2: Data-Driven Kernel — Feature Importance (27 Atmospheric Features)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGDIR / "fig9_step2_vip.pdf")
    plt.close(fig)
    print("  fig9_step2_vip.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 10: Step 2 — PLS Diagnostics
# ══════════════════════════════════════════════════════════════
def fig10_step2_diagnostics(step2_data):
    """2x2 diagnostic panel for Step 2: score plot, Hotelling T², DModX, loading plot."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    kernel = step2_data["kernel"]
    pls = kernel.pls_model_
    r = pls.results_
    X_test = step2_data["X_test_c"]
    feature_names = step2_data["feature_names"]
    n_feat = len(feature_names)

    # Compute test-set scores
    T_test = pls.transform(X_test)
    T_train = r.x_scores

    # ── (a) Score plot: t1 vs t2 ──
    ax = axes[0, 0]
    n_plot = min(5000, len(T_test))
    rng = np.random.RandomState(42)
    idx = rng.choice(len(T_test), n_plot, replace=False)

    # Color by T_surface anomaly magnitude (proxy for climate state)
    X_test_full = step2_data["X_test_c"]
    t_surf_idx = feature_names.index("T_surface") if "T_surface" in feature_names else -2
    t_surf_anom = X_test_full[idx, t_surf_idx]

    scatter = ax.scatter(
        T_test[idx, 0], T_test[idx, 1],
        c=t_surf_anom, cmap="coolwarm", s=4, alpha=0.4,
        vmin=-np.percentile(np.abs(t_surf_anom), 95),
        vmax=np.percentile(np.abs(t_surf_anom), 95),
    )
    plt.colorbar(scatter, ax=ax, label="$\\Delta T_{surface}$ (K)", shrink=0.8)
    ax.set_xlabel("$t_1$ (Score 1)")
    ax.set_ylabel("$t_2$ (Score 2)")
    ax.set_title("(a) Score Plot Colored by $\\Delta T_{surface}$")
    ax.axhline(0, color="#999", ls=":", lw=0.5)
    ax.axvline(0, color="#999", ls=":", lw=0.5)

    # Annotate variance explained
    x_var = r.x_variance_explained
    if x_var is not None and len(x_var) >= 2:
        ax.set_xlabel(f"$t_1$ ({x_var[0]*100:.1f}% X-var)")
        ax.set_ylabel(f"$t_2$ ({x_var[1]*100:.1f}% X-var)")

    # ── (b) Hotelling's T² ──
    ax = axes[0, 1]
    T2_test = _compute_hotellings_t2(T_test)
    A = T_train.shape[1]
    from scipy import stats
    T2_crit_95 = stats.chi2.ppf(0.95, A) * A

    ax.hist(T2_test, bins=80, density=True, color="#42a5f5", alpha=0.7,
            edgecolor="white", linewidth=0.3, label="Test set")
    ax.axvline(T2_crit_95, color="#c62828", ls="--", lw=1.5,
               label=f"95% limit ({T2_crit_95:.1f})")
    pct_above = 100 * np.mean(T2_test > T2_crit_95)
    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("Density")
    ax.set_title(f"(b) Hotelling's $T^2$ ({pct_above:.1f}% above 95% limit)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, np.percentile(T2_test, 99.5))

    # ── (c) DModX ──
    ax = axes[1, 0]
    x_mean = r.x_mean if r.x_mean is not None else step2_data["X_mean"]
    P = r.x_loadings
    dmodx_test = _compute_dmodx(X_test + step2_data["X_mean"], T_test, P, x_mean)

    # Color by cloud fraction anomaly
    cf_idx = feature_names.index("cloud_fraction") if "cloud_fraction" in feature_names else -1
    cf_anom = X_test_full[idx, cf_idx]

    ax.scatter(np.arange(n_plot), dmodx_test[idx], c=cf_anom, cmap="PuOr",
               s=4, alpha=0.4,
               vmin=-np.percentile(np.abs(cf_anom), 95),
               vmax=np.percentile(np.abs(cf_anom), 95))

    # D-crit: mean + 2*std of training DModX
    X_train_raw = step2_data["X_train_c"] + step2_data["X_mean"]
    dmodx_train = _compute_dmodx(X_train_raw, T_train, P, x_mean)
    dcrit = np.mean(dmodx_train) + 2 * np.std(dmodx_train)
    ax.axhline(dcrit, color="#c62828", ls="--", lw=1.2,
               label=f"D-crit ({dcrit:.2f})")
    pct_above_d = 100 * np.mean(dmodx_test > dcrit)
    ax.set_xlabel("Observation index (subsampled)")
    ax.set_ylabel("DModX (RMS X-residual)")
    ax.set_title(f"(c) DModX ({pct_above_d:.1f}% above D-crit)")
    ax.legend(fontsize=8)

    # ── (d) Loading plot: p1 vs p2 with feature labels ──
    ax = axes[1, 1]

    def _feat_color(name):
        if name.startswith("q_"):
            return "#26a69a"
        elif name == "cloud_fraction":
            return "#ff7043"
        elif name == "T_surface":
            return "#ffa726"
        else:
            return "#42a5f5"

    feat_colors = [_feat_color(n) for n in feature_names]

    ax.scatter(P[:n_feat, 0], P[:n_feat, 1], s=40, c=feat_colors,
               edgecolor="#333", linewidth=0.5, zorder=5)

    # Label placement: use adjustText if available, otherwise manual
    for j in range(n_feat):
        # Shorten labels for readability
        short = feature_names[j].replace("_", " ")
        ax.annotate(
            short, xy=(P[j, 0], P[j, 1]),
            fontsize=6, ha="center", va="bottom",
            xytext=(0, 4), textcoords="offset points",
        )

    # Reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    r_max = max(np.max(np.abs(P[:n_feat, 0])), np.max(np.abs(P[:n_feat, 1]))) * 1.1
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta),
            color="#ccc", ls=":", lw=0.8)

    if x_var is not None and len(x_var) >= 2:
        ax.set_xlabel(f"$p_1$ ({x_var[0]*100:.1f}% X-var)")
        ax.set_ylabel(f"$p_2$ ({x_var[1]*100:.1f}% X-var)")
    else:
        ax.set_xlabel("$p_1$ (Loading 1)")
        ax.set_ylabel("$p_2$ (Loading 2)")
    ax.set_title("(d) Loading Plot (27 Atmospheric Features)")
    ax.axhline(0, color="#999", ls=":", lw=0.5)
    ax.axvline(0, color="#999", ls=":", lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")

    for a in axes.flat:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.suptitle("Step 2: Data-Driven Kernel — PLS Diagnostics (10 Components, 27 Features)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGDIR / "fig10_step2_diagnostics.pdf")
    plt.close(fig)
    print("  fig10_step2_diagnostics.pdf")


# ══════════════════════════════════════════════════════════════
# Figure 11: Step 2 — Component Sweep
# ══════════════════════════════════════════════════════════════
def fig11_step2_component_sweep(step2_data):
    """R²/Q² vs components for Step 2 (27 features)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    X_train = step2_data["X_train_c"]
    X_test = step2_data["X_test_c"]
    Y_train_c = step2_data["Y_train"] - step2_data["Y_mean"]
    Y_test = step2_data["Y_test"]
    Y_mean = step2_data["Y_mean"]

    max_comp = min(20, X_train.shape[1])
    comps = list(range(1, max_comp + 1))
    r2_train, q2_test = [], []

    for nc in comps:
        pls = ConstrainedNipalsPLS(n_components=nc)
        pls.fit(X_train, Y_train_c)

        Y_pred_tr = pls.predict(X_train)
        ss_res_tr = np.sum((Y_train_c - Y_pred_tr) ** 2)
        ss_tot_tr = np.sum((Y_train_c - np.mean(Y_train_c, axis=0)) ** 2)
        r2 = 1 - ss_res_tr / ss_tot_tr

        Y_pred_te = pls.predict(X_test)
        ss_res_te = np.sum((Y_test - Y_mean - Y_pred_te) ** 2)
        ss_tot_te = np.sum((Y_test - np.mean(Y_test, axis=0)) ** 2)
        q2 = 1 - ss_res_te / ss_tot_te

        r2_train.append(r2)
        q2_test.append(q2)

    # ── Left: R² and Q² vs components ──
    ax1.plot(comps, r2_train, "s-", color=C_RETUNE_B, markersize=5,
             label="Train $R^2$", linewidth=1.5)
    ax1.plot(comps, q2_test, "o-", color="#42a5f5", markersize=5,
             label="Test $Q^2$", linewidth=1.5)
    ax1.axvline(10, color="#999", ls=":", lw=1, label="$A=10$ (used)")
    ax1.set_xlabel("Number of PLS Components ($A$)")
    ax1.set_ylabel("$R^2$ / $Q^2$")
    ax1.set_title("Step 2 Performance vs Components")
    ax1.set_xticks(comps)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate key values
    for i, nc in enumerate(comps):
        if nc in (1, 3, 5, 10, 15, 20):
            ax1.annotate(f"{q2_test[i]:.3f}", xy=(nc, q2_test[i]),
                         xytext=(5, 8), textcoords="offset points",
                         fontsize=7, color="#42a5f5")

    # ── Right: Marginal Q² gain per component ──
    delta_q2 = [q2_test[i] - q2_test[i - 1] for i in range(1, len(comps))]
    bar_colors = ["#42a5f5" if i < 10 else C_KERNEL for i in range(len(delta_q2))]
    ax2.bar(comps[1:], delta_q2, color=bar_colors, edgecolor="white", width=0.6)
    ax2.axhline(0.001, color="#c62828", ls="--", lw=1, label="$\\Delta Q^2 = 0.001$")
    ax2.axhline(0, color="#999", ls=":", lw=0.5)
    ax2.set_xlabel("Component Added")
    ax2.set_ylabel("$\\Delta Q^2$")
    ax2.set_title("Marginal $Q^2$ Gain per Component")
    ax2.set_xticks(comps[1:])
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for i in range(1, len(comps)):
        dq = q2_test[i] - q2_test[i - 1]
        if abs(dq) > 0.002 or i in (2, 4, 9, 14, 19):
            ax2.text(comps[i], dq + 0.001, f"{dq:+.4f}", ha="center",
                     va="bottom", fontsize=6, rotation=45)

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig11_step2_component_sweep.pdf")
    plt.close(fig)
    print("  fig11_step2_component_sweep.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    print("Generating ClimKern-Retune paper figures...")
    print(f"Output: {FIGDIR}")
    print()

    # Generate data and fit pipeline
    print("Fitting two-stage pipeline on CERES+NCEP real data...")
    harmonizer, dashboard, X_train, X_test, Y_train, Y_test, mk_data = fit_pipeline_real()
    print(f"  Best method: {dashboard.best_method} (Q² = {dashboard.best_q2:.4f})")
    print()

    # Generate all figures
    print("Generating figures:")
    fig1_pipeline()
    fig2_q2_dashboard(dashboard)
    fig3_regime_q2(dashboard)
    fig4_retune_comparison(dashboard)
    fig5_spread_reduction(dashboard, harmonizer, X_test, Y_test)
    fig6_kernel_weights(harmonizer)
    fig7_diagnostics(harmonizer, X_test, Y_test)
    fig7b_component_sweep(X_train, X_test, Y_train, Y_test)

    # Step 2: Data-driven kernel diagnostics
    print()
    print("Fitting Step 2 data-driven kernel on CERES+NCEP real data...")
    step2_data = fit_step2_real()
    if step2_data is not None:
        step2_kernel = step2_data["kernel"]
        step2_q2 = step2_kernel.pls_model_.results_.y_variance_explained
        print(f"  Step 2 fitted with 10 components, {len(step2_data['feature_names'])} features")
        print()
        print("Generating Step 2 figures:")
        fig9_step2_vip(step2_data)
        fig10_step2_diagnostics(step2_data)
        fig11_step2_component_sweep(step2_data)
    else:
        print("  Step 2 data not available, skipping Step 2 figures")

    print()
    print(f"Done! {len(list(FIGDIR.glob('*.pdf')))} figures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
