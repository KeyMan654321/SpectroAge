"""
SpectroAge — Plots
==================
Publication-quality plots for the benchmark exhibit.

All figures designed for an ISEF display board:
  - High DPI (180)
  - Clean, minimal style
  - Colour-blind-friendly palette
  - Large readable fonts

Usage
-----
    from spectroage_plots import SpectroAgePlots
    plotter = SpectroAgePlots(output_dir="figures")
    plotter.plot_all(y_true, y_pred, sigmas, cv_results, calibration)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Colour palette (colourblind-friendly) ────────────────────────────────────
TEAL    = "#1D9E75"
PURPLE  = "#7F77DD"
AMBER   = "#BA7517"
CORAL   = "#D85A30"
GRAY    = "#888780"
DARK    = "#2C2C2A"
LIGHT   = "#F1EFE8"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.facecolor":    LIGHT,
    "figure.facecolor":  "#FAFAF8",
    "axes.edgecolor":    GRAY,
    "axes.linewidth":    0.6,
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.9,
    "xtick.color":       DARK,
    "ytick.color":       DARK,
    "text.color":        DARK,
    "axes.labelcolor":   DARK,
    "axes.titlepad":     10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  GRAY,
})


class SpectroAgePlots:

    def __init__(self, output_dir: str = "figures"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    # ── 1. Predicted vs True ─────────────────────────────────────────────────

    def plot_pred_vs_true(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        sigmas:  np.ndarray,
        tool_name: str = "SpectroAge",
        save: bool = True,
    ) -> plt.Figure:
        """
        Scatter plot of predicted vs. true age with error bars,
        residual histogram, and residual vs. age trend.
        """
        residuals = y_pred - y_true
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r, _ = stats.pearsonr(y_true, y_pred)

        fig = plt.figure(figsize=(13, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        # ── Panel 1: scatter ─────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(
            y_true, y_pred, c=sigmas, cmap="YlOrRd",
            s=14, alpha=0.7, linewidths=0, zorder=3,
        )
        lim = (0, max(y_true.max(), y_pred.max()) * 1.05)
        ax1.plot(lim, lim, "--", color=GRAY, lw=1.2, zorder=2, label="1:1 line")
        ax1.set_xlim(lim); ax1.set_ylim(lim)
        ax1.set_xlabel("True age (Gyr)")
        ax1.set_ylabel("Predicted age (Gyr)")
        ax1.set_title(f"{tool_name}: Predicted vs. True")
        cb = fig.colorbar(sc, ax=ax1, shrink=0.75, pad=0.02)
        cb.set_label("1σ uncertainty (Gyr)", fontsize=8)
        ax1.text(0.04, 0.96,
                 f"MAE = {mae:.2f} Gyr\nRMSE = {rmse:.2f} Gyr\nr = {r:.3f}",
                 transform=ax1.transAxes, fontsize=8.5, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        # ── Panel 2: residual histogram ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(residuals, bins=40, color=TEAL, alpha=0.8,
                 edgecolor="white", linewidth=0.4)
        ax2.axvline(0, color=DARK, lw=1.2, linestyle="--")
        ax2.axvline(np.mean(residuals), color=CORAL, lw=1.5,
                    linestyle="-", label=f"mean bias = {np.mean(residuals):+.2f}")
        ax2.set_xlabel("Residual: pred − true (Gyr)")
        ax2.set_ylabel("Count")
        ax2.set_title("Age Residuals")
        ax2.legend()

        # Overlay Gaussian fit
        xg = np.linspace(residuals.min(), residuals.max(), 200)
        mu, sd = np.mean(residuals), np.std(residuals)
        n_bin  = len(residuals)
        bwidth = (residuals.max() - residuals.min()) / 40
        ax2.plot(xg, stats.norm.pdf(xg, mu, sd) * n_bin * bwidth,
                 color=CORAL, lw=1.5, label=f"Gaussian (σ={sd:.2f})")
        ax2.legend(fontsize=8)

        # ── Panel 3: residual vs. true age ────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(y_true, residuals, c=PURPLE, s=12,
                    alpha=0.5, linewidths=0, zorder=3)
        ax3.axhline(0, color=DARK, lw=1.2, linestyle="--")

        # Rolling median to show systematic trends
        order = np.argsort(y_true)
        yt_s  = y_true[order]
        rs_s  = residuals[order]
        window = max(len(y_true) // 20, 5)
        roll_med = np.convolve(rs_s, np.ones(window)/window, mode="valid")
        roll_x   = yt_s[window//2 : window//2 + len(roll_med)]
        ax3.plot(roll_x, roll_med, color=AMBER, lw=2, label="Rolling median")

        ax3.set_xlabel("True age (Gyr)")
        ax3.set_ylabel("Residual (Gyr)")
        ax3.set_title("Systematic Trends")
        ax3.legend(fontsize=8)

        fig.suptitle("SpectroAge — Accuracy Assessment",
                     fontsize=13, fontweight="bold", y=1.01)

        if save:
            fp = self.out / "01_pred_vs_true.png"
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  Saved {fp}")
        return fig

    # ── 2. Uncertainty calibration ───────────────────────────────────────────

    def plot_calibration(
        self,
        y_true:  np.ndarray,
        y_pred:  np.ndarray,
        sigmas:  np.ndarray,
        save: bool = True,
    ) -> plt.Figure:
        """
        Reliability diagram: expected vs. observed coverage at each
        confidence level. A perfectly calibrated model lies on the diagonal.
        """
        levels = np.linspace(0.05, 0.99, 40)
        observed = []
        for lv in levels:
            z = stats.norm.ppf((1 + lv) / 2)
            covered = np.mean(np.abs(y_pred - y_true) < z * sigmas)
            observed.append(covered)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        ax1.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Perfect calibration")
        ax1.plot(levels, observed, color=TEAL, lw=2.0, label="SpectroAge")
        ax1.fill_between(levels, levels, observed,
                         alpha=0.15, color=TEAL)
        ax1.set_xlabel("Expected coverage")
        ax1.set_ylabel("Observed coverage")
        ax1.set_title("Calibration Reliability Diagram")
        ax1.legend()
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

        # Panel 2: uncertainty vs. absolute error
        ax2.scatter(sigmas, np.abs(y_pred - y_true),
                    c=PURPLE, s=12, alpha=0.4, linewidths=0)
        max_val = max(sigmas.max(), np.abs(y_pred - y_true).max())
        ax2.plot([0, max_val], [0, max_val], "--", color=GRAY, lw=1.2,
                 label="σ = |error| (ideal)")
        ax2.set_xlabel("Predicted σ (Gyr)")
        ax2.set_ylabel("|Predicted − True| (Gyr)")
        ax2.set_title("Uncertainty vs. Actual Error")
        ax2.legend(fontsize=8)

        fig.suptitle("SpectroAge — Uncertainty Calibration",
                     fontsize=13, fontweight="bold")

        if save:
            fp = self.out / "02_calibration.png"
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  Saved {fp}")
        return fig

    # ── 3. Feature importance ────────────────────────────────────────────────

    def plot_feature_importance(
        self,
        sa,          # trained SpectroAge instance
        X: np.ndarray,
        y: np.ndarray,
        save: bool = True,
    ) -> plt.Figure:
        """
        Permutation importance: how much does MAE increase when each
        feature is randomly shuffled? Larger = more important.
        """
        from spectroage import FEATURE_NAMES
        baseline_pred, _ = sa.predict(X)
        baseline_mae = mean_absolute_error(y, baseline_pred)

        importances = []
        rng = np.random.default_rng(SEED := 42)
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            pred_perm, _ = sa.predict(X_perm)
            delta_mae = mean_absolute_error(y, pred_perm) - baseline_mae
            importances.append(delta_mae)

        importances = np.array(importances)
        order = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [TEAL if imp > 0 else CORAL for imp in importances[order]]
        bars = ax.barh(
            [FEATURE_NAMES[i] for i in order],
            importances[order],
            color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        ax.axvline(0, color=DARK, lw=0.8)
        ax.set_xlabel("Increase in MAE when feature is shuffled (Gyr)")
        ax.set_title("Feature Importance (Permutation Method)")
        ax.invert_yaxis()

        for bar, val in zip(bars, importances[order]):
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=8,
            )

        fig.tight_layout()
        if save:
            fp = self.out / "03_feature_importance.png"
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  Saved {fp}")
        return fig

    # ── 4. Age distribution ──────────────────────────────────────────────────

    def plot_age_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sigmas: np.ndarray,
        save:   bool = True,
    ) -> plt.Figure:
        """
        Overlay predicted and true age distributions with uncertainty bands.
        """
        fig, ax = plt.subplots(figsize=(8, 4))

        bins = np.linspace(0, 14, 35)
        ax.hist(y_true, bins=bins, alpha=0.5, color=GRAY,
                label="True ages", edgecolor="white", linewidth=0.3)
        ax.hist(y_pred, bins=bins, alpha=0.6, color=TEAL,
                label="SpectroAge estimates", edgecolor="white", linewidth=0.3)

        # Shade ±1σ band around prediction distribution
        counts_pred, edges = np.histogram(y_pred, bins=bins)
        ax.bar(edges[:-1], counts_pred, width=np.diff(edges),
               alpha=0.0, align="edge")  # invisible, just for spacing

        ax.set_xlabel("Stellar age (Gyr)")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution: True vs. SpectroAge Predicted")
        ax.legend()

        mean_sigma = np.mean(sigmas)
        ax.text(0.97, 0.95,
                f"Mean σ = {mean_sigma:.2f} Gyr",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        fig.tight_layout()
        if save:
            fp = self.out / "04_age_distribution.png"
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  Saved {fp}")
        return fig

    # ── 5. HR-diagram coloured by predicted age ───────────────────────────────

    def plot_hr_diagram(
        self,
        X:      np.ndarray,
        y_pred: np.ndarray,
        sigmas: np.ndarray,
        save:   bool = True,
    ) -> plt.Figure:
        """
        Hertzsprung-Russell diagram (Teff vs. logg) coloured by
        SpectroAge-predicted stellar age.
        """
        Teff = X[:, 0]
        logg = X[:, 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: coloured by age
        sc1 = ax1.scatter(Teff, logg, c=y_pred, cmap="plasma_r",
                          s=10, alpha=0.6, linewidths=0,
                          vmin=0, vmax=13)
        ax1.invert_xaxis()
        ax1.invert_yaxis()
        ax1.set_xlabel("Effective Temperature Teff (K)")
        ax1.set_ylabel("Surface Gravity log g")
        ax1.set_title("HR Diagram — Coloured by Predicted Age")
        cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.85)
        cb1.set_label("Predicted age (Gyr)", fontsize=8)
        ax1.text(0.97, 0.03, "← Hotter    Cooler →",
                 transform=ax1.transAxes, ha="right", fontsize=7.5,
                 color=GRAY)

        # Panel 2: coloured by uncertainty
        sc2 = ax2.scatter(Teff, logg, c=sigmas, cmap="YlOrRd",
                          s=10, alpha=0.6, linewidths=0)
        ax2.invert_xaxis()
        ax2.invert_yaxis()
        ax2.set_xlabel("Effective Temperature Teff (K)")
        ax2.set_ylabel("Surface Gravity log g")
        ax2.set_title("HR Diagram — Coloured by Uncertainty")
        cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.85)
        cb2.set_label("1σ uncertainty (Gyr)", fontsize=8)

        fig.suptitle("SpectroAge — Stellar Parameter Space",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

        if save:
            fp = self.out / "05_hr_diagram.png"
            fig.savefig(fp, dpi=180, bbox_inches="tight")
            print(f"  Saved {fp}")
        return fig

    # ── All plots ─────────────────────────────────────────────────────────────

    def plot_all(
        self,
        sa,
        X_test:  np.ndarray,
        y_test:  np.ndarray,
        y_pred:  np.ndarray,
        sigmas:  np.ndarray,
    ) -> None:
        print("\nGenerating all figures...")
        self.plot_pred_vs_true(y_test, y_pred, sigmas)
        self.plot_calibration(y_test, y_pred, sigmas)
        self.plot_feature_importance(sa, X_test[:200], y_test[:200])
        self.plot_age_distribution(y_test, y_pred, sigmas)
        self.plot_hr_diagram(X_test, y_pred, sigmas)
        print(f"\nAll figures saved to {self.out}/")
