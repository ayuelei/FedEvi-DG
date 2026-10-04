#!/usr/bin/env python3
"""Generate reproducible manuscript figures from frozen FedEvi-DG outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
FINAL_ANALYSIS = ROOT / "outputs" / "fedevi_final_analysis_v3_audit"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "Noto Sans CJK JP"
COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#6B7280",
    "light": "#F4F7FA",
    "ink": "#17212C",
}

mpl.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, text, facecolor, edgecolor=None, fontsize=8, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.15,
        edgecolor=edgecolor or facecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, color=None, dashed=False, rad=0.0, width=1.25):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=width,
        linestyle="--" if dashed else "-",
        color=color or COLORS["gray"],
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def figure_framework() -> None:
    fig = plt.figure(figsize=(7.15, 4.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.35])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax1.text(0.00, 1.02, "A", fontsize=12, weight="bold", transform=ax1.transAxes)
    ax1.text(0.50, 0.98, "异构临床标注与联邦协作", ha="center", va="top", weight="bold", fontsize=10)
    client_y = [0.79, 0.64, 0.49, 0.34, 0.19]
    clients = [
        ("BUSBRA", "分类 + 掩码 + BI-RADS", COLORS["sky"]),
        ("BUSI_WHU", "分类 + 掩码", COLORS["green"]),
        ("GDPH", "分类 + BI-RADS", COLORS["orange"]),
        ("SYSUCC", "分类 + BI-RADS", COLORS["orange"]),
        ("UDIAT", "分类 + 掩码", COLORS["green"]),
    ]
    for y, (name, labels, color) in zip(client_y, clients, strict=True):
        box(ax1, (0.02, y), 0.40, 0.095, f"{name}\n{labels}", "white", color, 7.1, "bold")
        arrow(ax1, (0.43, y + 0.048), (0.60, 0.55), color=COLORS["gray"], width=0.9)
    box(ax1, (0.61, 0.43), 0.36, 0.24, "中央服务器\n\n共享参数：按样本数聚合\n辅助分支：按有效标注数聚合", "#E8EEF4", COLORS["blue"], 7.2, "bold")
    ax1.text(0.50, 0.08, "仅交换模型参数；目标域不参与训练与选择", ha="center", fontsize=7.5, color=COLORS["gray"])
    ax1.plot([0.02, 0.97], [0.13, 0.13], color="#D5DCE3", lw=0.8)

    ax2.text(0.00, 1.02, "B", fontsize=12, weight="bold", transform=ax2.transAxes)
    ax2.text(0.50, 0.98, "病灶引导特征池化与非负系数约束的序数评分融合", ha="center", va="top", weight="bold", fontsize=9.4)
    box(ax2, (0.02, 0.62), 0.16, 0.14, "乳腺超声\n图像 x", "#E8EEF4", COLORS["blue"], 8, "bold")
    box(ax2, (0.25, 0.62), 0.18, 0.14, "ResNet18\n特征图 F(x)", "#DDEBF7", COLORS["blue"], 8, "bold")
    box(ax2, (0.50, 0.70), 0.19, 0.13, "病灶分割分支\n概率图 M(x)", "#DDF2EA", COLORS["green"], 7.7, "bold")
    box(ax2, (0.50, 0.48), 0.19, 0.13, "病灶概率加权池化\n特征 zL", "#DDF2EA", COLORS["green"], 7.5, "bold")
    box(ax2, (0.77, 0.60), 0.20, 0.13, "序数回归分支\n评分 r∈[0,1]", "#FFF0D4", COLORS["orange"], 7.7, "bold")
    box(ax2, (0.51, 0.23), 0.45, 0.13, "序数评分融合（非负系数）\nℓ = b(zL) + softplus(a)·r", "#EEE7F4", COLORS["purple"], 8.1, "bold")
    box(ax2, (0.63, 0.04), 0.22, 0.11, "恶性概率\nP(y=1|x)", "#FADFD4", COLORS["vermillion"], 8, "bold")
    arrow(ax2, (0.18, 0.69), (0.25, 0.69), COLORS["blue"])
    arrow(ax2, (0.43, 0.69), (0.50, 0.765), COLORS["green"])
    arrow(ax2, (0.595, 0.70), (0.595, 0.61), COLORS["green"])
    arrow(ax2, (0.69, 0.545), (0.77, 0.665), COLORS["orange"])
    arrow(ax2, (0.595, 0.48), (0.63, 0.36), COLORS["green"])
    arrow(ax2, (0.87, 0.60), (0.84, 0.36), COLORS["orange"])
    arrow(ax2, (0.735, 0.23), (0.735, 0.15), COLORS["vermillion"])
    ax2.annotate("人工标注掩码监督\n（仅标注可用样本）", xy=(0.595, 0.83), xytext=(0.46, 0.91), fontsize=7, color=COLORS["green"], ha="center")
    ax2.annotate("BI-RADS 等级监督\n（仅标注可用样本）", xy=(0.87, 0.73), xytext=(0.84, 0.88), fontsize=7, color=COLORS["orange"], ha="center")
    ax2.text(0.03, 0.40, "推理只输入图像\n不需要人工掩码或 BI-RADS", fontsize=7.6, color=COLORS["gray"], ha="left")
    save(fig, "figure1_framework")


def graphical_abstract() -> None:
    """One-page visual summary for the Chinese manuscript front matter."""
    fig, ax = plt.subplots(figsize=(10.8, 3.25), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "FedEvi-DG：异构临床标注下的乳腺超声联邦域泛化",
        ha="center",
        va="top",
        fontsize=15,
        weight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.865,
        "训练时利用不同中心可获得的病灶掩码与 BI-RADS；推理时只输入超声图像",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["gray"],
    )

    box(ax, (0.025, 0.30), 0.155, 0.38,
        "五个开发域\n\n分类标签：全部\n病灶掩码：部分\nBI-RADS：部分",
        "#E8EEF4", COLORS["blue"], 9, "bold")
    box(ax, (0.225, 0.30), 0.155, 0.38,
        "联邦训练\n\n原始图像保留在本地\n辅助分支仅使用\n可用人工标注监督",
        "#DDEBF7", COLORS["blue"], 9, "bold")
    box(ax, (0.425, 0.30), 0.165, 0.38,
        "病灶引导特征\n\n病灶概率图 M(x)\n＋\n加权池化特征 zL",
        "#DDF2EA", COLORS["green"], 9, "bold")
    box(ax, (0.635, 0.30), 0.155, 0.38,
        "序数评分融合\n（非负系数）\n\nℓ=b(zL)+softplus(α)·r\n\n固定病灶特征时\n评分上升不降低 logit",
        "#FFF0D4", COLORS["orange"], 8.7, "bold")
    box(ax, (0.835, 0.30), 0.14, 0.38,
        "验证\n\n集成 AUROC\n五域 LODO：0.860\nQAMEBI：0.854\n外部差异不确定",
        "#FADFD4", COLORS["vermillion"], 8.7, "bold")
    for start, end, color in [
        ((0.18, 0.49), (0.225, 0.49), COLORS["blue"]),
        ((0.38, 0.49), (0.425, 0.49), COLORS["green"]),
        ((0.59, 0.49), (0.635, 0.49), COLORS["orange"]),
        ((0.79, 0.49), (0.835, 0.49), COLORS["vermillion"]),
    ]:
        arrow(ax, start, end, color=color, width=1.6)
    ax.text(
        0.5,
        0.12,
        "结论边界：五域 LODO 结果支持整体模型；独立公开外部数据未证明其优于 FedAvg，仍需患者级独立临床验证",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F7F8FA", "edgecolor": "#CBD3DB"},
    )
    save(fig, "graphical_abstract")


def figure_internal_results() -> None:
    """Backward-compatible entry point for the standalone internal-result figures."""
    figure_internal_results_v2()


def figure_external_evidence() -> None:
    """Backward-compatible entry point for the external-evidence figures."""
    figure_external_evidence_v2()


def figure_clinical_utility() -> None:
    """Backward-compatible entry point for the clinical-utility figures."""
    figure_clinical_utility_v2()


def compose_figure_from_pngs(stem: str, panels: list[list[str]], figsize: tuple[float, float]) -> None:
    """Compose already-rendered panels into one manuscript figure without redrawing data."""
    rows = len(panels)
    cols = max(len(row) for row in panels)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(rows, cols)
    for row_index, row in enumerate(panels):
        for col_index, panel_stem in enumerate(row):
            slot = grid[row_index, :] if len(row) == 1 else grid[row_index, col_index]
            ax = fig.add_subplot(slot)
            ax.imshow(plt.imread(OUT / f"{panel_stem}.png"))
            ax.axis("off")
    save(fig, stem)


METHOD_LABELS = {
    "fedavg": "FedAvg",
    "fedevi_mtl": "Multitask baseline",
    "fedevi_seg_only": "Classification + segmentation",
    "fedavg_ord": "Classification + ordinal",
    "fedevi_mask_shift": "Shifted lesion masks",
    "fedevi_shuffle": "Permuted BI-RADS labels",
    "fedevi_dg": "Unconstrained ordinal fusion",
    "fedevi_no_taskagg": "All-client auxiliary aggregation",
    "fedevi_monotonic": "FedEvi-DG",
}


def figure_internal_results_v2() -> None:
    """Export heatmap, seed stability, and paired effects as standalone figures."""
    ensemble = pd.read_csv(FINAL_ANALYSIS / "ensemble_metrics.csv")
    per_seed = pd.read_csv(FINAL_ANALYSIS / "per_seed_auc.csv")
    boot = pd.read_csv(FINAL_ANALYSIS / "paired_bootstrap_macro.csv")
    domains = ["BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT"]
    methods = [
        "fedevi_monotonic", "fedevi_no_taskagg", "fedevi_shuffle",
        "fedevi_dg", "fedevi_mtl", "fedevi_seg_only", "fedavg",
        "fedevi_mask_shift", "fedavg_ord",
    ]
    fig, ax_heat = plt.subplots(figsize=(7.15, 4.15), constrained_layout=True)
    matrix = (
        ensemble.pivot(index="method", columns="target", values="auc")
        .loc[methods, domains]
    )
    image = ax_heat.imshow(matrix.to_numpy(), cmap="cividis", vmin=0.75, vmax=0.94, aspect="auto")
    ax_heat.set_xticks(np.arange(len(domains)), domains)
    ax_heat.set_yticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods])
    ax_heat.set_xlabel("Completely unseen target domain")
    ax_heat.set_title("A　Ensemble AUROC across five leave-one-domain-out rotations", loc="left", weight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            rgba = image.cmap(image.norm(float(value)))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.5 else COLORS["ink"]
            ax_heat.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7.4, color=color,
                         weight="bold" if methods[row] == "fedevi_monotonic" else "normal")
    ax_heat.set_xticks(np.arange(-0.5, len(domains), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.6)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("AUROC")
    save(fig, "figure2A_lodo_heatmap")

    fig, ax_seed = plt.subplots(figsize=(6.6, 3.55), constrained_layout=True)
    macro_seed = per_seed.groupby(["method", "seed"], as_index=False).auc.mean()
    summary = macro_seed.groupby("method").auc.agg(["mean", "std"]).loc[methods]
    plot_methods = methods[::-1]
    y = np.arange(len(plot_methods))
    for yi, method in zip(y, plot_methods, strict=True):
        color = COLORS["vermillion"] if method == "fedevi_monotonic" else COLORS["gray"]
        ax_seed.errorbar(
            float(summary.loc[method, "mean"]),
            yi,
            xerr=float(summary.loc[method, "std"]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3.0,
            lw=1.35,
            ms=5.2,
            zorder=3,
        )
    ax_seed.set_yticks(y, [METHOD_LABELS[m] for m in plot_methods])
    ax_seed.set_xlabel("Five-domain macro AUROC (mean ± SD)")
    ax_seed.set_xlim(0.812, 0.862)
    ax_seed.set_title("B　Three-seed stability", loc="left", weight="bold")
    ax_seed.grid(axis="x", color="#E2E7EB", lw=0.6)
    save(fig, "figure2B_seed_stability")

    fig, ax_forest = plt.subplots(figsize=(5.8, 3.9), constrained_layout=True)
    order = [
        "fedavg", "fedevi_mtl", "fedevi_seg_only", "fedavg_ord",
        "fedevi_mask_shift", "fedevi_shuffle", "fedevi_dg", "fedevi_no_taskagg",
    ]
    frame = boot.set_index("comparator").loc[order].iloc[::-1]
    y = np.arange(len(frame))
    xval = frame.macro_auc_delta.to_numpy()
    lo = frame.ci95_lower.to_numpy()
    hi = frame.ci95_upper.to_numpy()
    ax_forest.axvline(0, color=COLORS["ink"], lw=0.9, ls=":")
    for yi, xi, low, high in zip(y, xval, lo, hi, strict=True):
        supported = low > 0
        color = COLORS["vermillion"] if supported else COLORS["gray"]
        marker = "o" if supported else "D"
        ax_forest.errorbar(xi, yi, xerr=[[xi-low], [high-xi]], fmt=marker, color=color,
                           ecolor=color, capsize=2.5, ms=4.5, lw=1.2)
    ax_forest.set_yticks(y, [METHOD_LABELS[m] for m in frame.index])
    ax_forest.set_xlabel("Macro AUROC difference (95% CI)\nFedEvi-DG − comparator")
    ax_forest.set_xlim(-0.011, 0.04)
    ax_forest.set_title("Paired bootstrap within domains", loc="left", weight="bold")
    ax_forest.grid(axis="x", color="#E2E7EB", lw=0.6)
    save(fig, "figure3_paired_bootstrap")

    # Descriptive cross-domain performance and seed stability form Figure 2;
    # the inferential paired-effect plot is promoted to standalone Figure 3.
    compose_figure_from_pngs(
        "figure2_performance_stability_combined",
        [["figure2A_lodo_heatmap"], ["figure2B_seed_stability"]],
        figsize=(7.15, 7.45),
    )


def figure_clinical_utility_v2() -> None:
    """Export Brier, ECE, and fixed-range DCA as standalone figures."""
    calibration = pd.read_csv(FINAL_ANALYSIS / "calibration.csv")
    dca = pd.read_csv(FINAL_ANALYSIS / "decision_curve.csv")
    dca = dca[dca.threshold.between(0.05, 0.30)].copy()
    domains = ["BUSBRA", "BUSI_WHU", "GDPH", "SYSUCC", "UDIAT"]
    def slope_panel(ax, raw_col, calibrated_col, ylabel, panel):
        positions = {"fedavg": (0, 1), "fedevi_monotonic": (2.2, 3.2)}
        colors = {"fedavg": COLORS["gray"], "fedevi_monotonic": COLORS["vermillion"]}
        for method in ("fedavg", "fedevi_monotonic"):
            frame = calibration[calibration.method.eq(method)].set_index("target").loc[domains]
            left, right = positions[method]
            for _, row in frame.iterrows():
                ax.plot([left, right], [row[raw_col], row[calibrated_col]], color=colors[method], alpha=0.24, lw=0.9)
                ax.scatter([left, right], [row[raw_col], row[calibrated_col]], color=colors[method], alpha=0.45, s=9)
            means = frame[[raw_col, calibrated_col]].mean().to_numpy()
            ax.plot([left, right], means, color=colors[method], lw=2.4, marker="o", ms=4.5)
        ax.set_xticks([0, 1, 2.2, 3.2], ["Before", "After", "Before", "After"])
        ax.text(0.5, -0.22, "FedAvg", transform=ax.get_xaxis_transform(), ha="center", color=COLORS["gray"], weight="bold")
        ax.text(2.7, -0.22, "FedEvi-DG", transform=ax.get_xaxis_transform(), ha="center", color=COLORS["vermillion"], weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E2E7EB", lw=0.6)
        ax.set_title(f"{panel}　Temperature scaling: {ylabel}", loc="left", weight="bold")

    fig, ax_brier = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)
    slope_panel(ax_brier, "raw_brier", "calibrated_brier", "Brier score", "A")
    save(fig, "figure4A_brier_calibration")

    fig, ax_ece = plt.subplots(figsize=(4.4, 3.35), constrained_layout=True)
    slope_panel(ax_ece, "raw_ece", "calibrated_ece", "ECE", "B")
    save(fig, "figure4B_ece_calibration")

    # Five domains in a 2x3 grid (last cell hidden) instead of one very wide
    # strip, so the standalone decision-curve figure stays legible.
    fig, dca_axes = plt.subplots(2, 3, figsize=(7.15, 4.6), sharey=True, constrained_layout=True)
    for ax in dca_axes.flat[len(domains):]:
        ax.set_visible(False)
    for index, (ax, domain) in enumerate(zip(dca_axes.flat[:len(domains)], domains, strict=True)):
        frame = dca[dca.target.eq(domain)]
        ax.plot(frame.threshold, frame.fedevi_monotonic, color=COLORS["vermillion"], lw=1.8, label="FedEvi-DG")
        ax.plot(frame.threshold, frame.fedavg, color=COLORS["blue"], lw=1.3, ls="--", label="FedAvg")
        ax.plot(frame.threshold, frame.treat_all, color=COLORS["gray"], lw=1.0, ls=":", label="Treat all")
        ax.axhline(0, color="#9AA4AE", lw=0.8, label="Treat none")
        ax.set_title(domain, fontsize=9, weight="bold", loc="left")
        ax.set_xlabel("Threshold probability")
        if index % 3 == 0:
            ax.set_ylabel("Net benefit")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(color="#EDF0F2", lw=0.45)
        ax.set_ylim(-0.015, 0.72)
    dca_axes.flat[0].legend(frameon=False, fontsize=7, loc="best")
    save(fig, "figure5_internal_decision_curves")

    # Figure 4 contains the two calibration panels; the decision curves are
    # presented as a separately named Figure 5 for legibility.
    compose_figure_from_pngs(
        "figure4_calibration_combined",
        [["figure4A_brier_calibration", "figure4B_ece_calibration"]],
        figsize=(7.15, 3.5),
    )


def figure_external_evidence_v2() -> None:
    """Export six standalone QAMEBI discrimination, utility, and evidence figures."""
    summary = json.loads((ROOT / "outputs/fedevi_qamebi_final_v1/summary/summary.json").read_text())
    evidence = json.loads((ROOT / "outputs/fedevi_qamebi_final_v1/summary/explanation_summary.json").read_text())
    predictions = pd.read_csv(ROOT / "outputs/fedevi_qamebi_final_v1/summary/ensemble_predictions.csv")
    reliability = pd.read_csv(ROOT / "outputs/fedevi_qamebi_final_v1/summary/clinical_utility/reliability.csv")
    dca = pd.read_csv(ROOT / "outputs/fedevi_qamebi_final_v1/summary/clinical_utility/decision_curve.csv")
    methods = ["fedavg", "fedevi_mtl", "fedevi_monotonic"]
    labels = [METHOD_LABELS[method] for method in methods]
    score_columns = ["score_fedavg", "score_fedevi_mtl", "score_fedevi_monotonic"]
    palette = [COLORS["gray"], COLORS["sky"], COLORS["vermillion"]]
    line_styles = ["--", "-.", "-"]
    fig, ax = plt.subplots(figsize=(3.45, 3.25), constrained_layout=True)
    for label, column, color, ls in zip(labels, score_columns, palette, line_styles, strict=True):
        fpr, tpr, _ = roc_curve(predictions.label, predictions[column])
        auc = summary["methods"][methods[labels.index(label)]]["ensemble_metrics"]["auc"]
        ax.plot(fpr, tpr, color=color, ls=ls, lw=1.7, label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#AAB2BA", ls=":", lw=0.9)
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("A　External ROC curves", loc="left", weight="bold")
    save(fig, "figure6A_external_roc")

    fig, ax = plt.subplots(figsize=(4.15, 2.85), constrained_layout=True)
    aucs = [summary["methods"][m]["ensemble_metrics"]["auc"] for m in methods]
    cis = [summary["methods"][m]["ensemble_auc_bootstrap_95_ci"] for m in methods]
    y = np.arange(3)[::-1]
    for yi, value, ci, color in zip(y, aucs, cis, palette, strict=True):
        ax.errorbar(value, yi, xerr=[[value-ci["lower"]], [ci["upper"]-value]], fmt="o", color=color,
                    ecolor=color, capsize=3, lw=1.4, ms=5)
        ax.text(ci["upper"] + 0.004, yi, f"{value:.3f}", va="center", fontsize=7)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.77, 0.92)
    ax.set_xlabel("AUROC (record-level 95% CI)")
    ax.set_title("B　Prespecified QAMEBI evaluation", loc="left", weight="bold")
    ax.grid(axis="x", color="#E2E7EB", lw=0.6)
    save(fig, "figure6B_external_auc")

    fig, ax = plt.subplots(figsize=(3.45, 3.25), constrained_layout=True)
    for method, label, color, ls in zip(methods, labels, palette, line_styles, strict=True):
        frame = reliability[reliability.method.eq(method) & reliability.n.gt(0)]
        ax.plot(frame.mean_predicted_probability, frame.observed_malignant_fraction,
                marker="o", ms=3.2, color=color, ls=ls, lw=1.3, label=label)
    ax.plot([0, 1], [0, 1], color="#AAB2BA", ls=":", lw=0.9)
    ax.set(xlabel="Predicted malignant probability", ylabel="Observed malignant fraction", xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("C　Calibration curve", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "figure6C_reliability")

    fig, ax = plt.subplots(figsize=(4.15, 3.0), constrained_layout=True)
    dca_styles = {
        "fedevi_monotonic": ("FedEvi-DG", COLORS["vermillion"], "-", 1.8),
        "fedavg": ("FedAvg", COLORS["blue"], "--", 1.3),
        "treat_all": ("Treat all", COLORS["gray"], ":", 1.0),
        "treat_none": ("Treat none", "#AAB2BA", "-.", 0.9),
    }
    for method, (label, color, ls, lw) in dca_styles.items():
        frame = dca[dca.method.eq(method) & dca.threshold_probability.between(0.05, 0.30)]
        ax.plot(frame.threshold_probability, frame.net_benefit, label=label, color=color, ls=ls, lw=lw)
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", xlim=(0.05, 0.30), ylim=(-0.02, 0.53))
    ax.set_title("D　External decision curves", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(color="#EDF0F2", lw=0.45)
    save(fig, "figure6D_external_dca")

    fig, ax = plt.subplots(figsize=(4.15, 3.0), constrained_layout=True)
    intervention_labels = ["Original image", "Predicted lesion\nretained", "Predicted lesion\nerased"]
    intervention_auc = [evidence["ensemble_original_auc"], evidence["ensemble_lesion_only_auc"], evidence["ensemble_lesion_erased_auc"]]
    bars = ax.bar(intervention_labels, intervention_auc, color=[COLORS["blue"], COLORS["green"], COLORS["orange"]], width=0.68)
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("AUROC")
    ax.set_title("A　Lesion-region perturbation", loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=0)
    for bar, value in zip(bars, intervention_auc, strict=True):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.004, f"{value:.3f}", ha="center", fontsize=7)
    ax.text(2, 0.792, "Δ = −0.082\n(vs. original)", ha="center", va="bottom",
            color=COLORS["vermillion"], fontsize=7)
    save(fig, "figure7A_lesion_intervention")

    fig, ax = plt.subplots(figsize=(4.3, 2.75), constrained_layout=True)
    metric_labels = ["Lesion Dice (seed mean)", "Lesion IoU (seed mean)", "Non-decreasing\nprobability fraction"]
    values = [evidence["seed_mean_lesion_dice"], evidence["seed_mean_lesion_iou"], evidence["ensemble_ordinal_non_decreasing_fraction"]]
    y = np.arange(3)[::-1]
    ax.barh(y, values, color=[COLORS["green"], COLORS["sky"], COLORS["purple"]], height=0.55)
    ax.set_yticks(y, metric_labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Metric value")
    ax.set_title("B　Localization & ordinal intervention", loc="left", weight="bold")
    for yi, value in zip(y, values, strict=True):
        ax.text(value+0.02, yi, f"{value:.3f}", va="center", fontsize=7)
    save(fig, "figure7B_localization_structure")

    # QAMEBI predictive performance and clinical utility form Figure 6.
    compose_figure_from_pngs(
        "figure6_qamebi_performance_combined",
        [
            ["figure6A_external_roc", "figure6B_external_auc"],
            ["figure6C_reliability", "figure6D_external_dca"],
        ],
        figsize=(7.15, 5.7),
    )

    # Input perturbations and ordinal-score intervention checks form Figure 7.
    compose_figure_from_pngs(
        "figure7_qamebi_evidence_combined",
        [["figure7A_lesion_intervention", "figure7B_localization_structure"]],
        figsize=(7.15, 3.35),
    )


if __name__ == "__main__":
    graphical_abstract()
    figure_framework()
    figure_internal_results_v2()
    figure_external_evidence_v2()
    figure_clinical_utility_v2()
    print(f"Generated manuscript figures in {OUT}")
