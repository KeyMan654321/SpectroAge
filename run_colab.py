"""
SpectroAge — Colab Quickstart
==============================
Run this file directly in Google Colab.
No arguments, no configuration needed.

Steps
-----
  1. Install dependencies
  2. Train SpectroAge on synthetic data (replace with real data)
  3. Evaluate and print metrics
  4. Generate and display all plots
  5. Save the trained model

To use real data, replace the `make_synthetic_training_data()` call
with your own CSV loader (see REAL DATA section below).
"""

# ── Step 0: Install dependencies ─────────────────────────────────────────────
import subprocess, sys

_PACKAGES = [
    "numpy", "scipy", "scikit-learn",
    "matplotlib", "pandas", "joblib",
]

for pkg in _PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg])

# ── Step 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

from spectroage import SpectroAge, make_synthetic_training_data, engineer_features
from spectroage_plots import SpectroAgePlots

print("SpectroAge — all modules loaded\n")

# ─────────────────────────────────────────────────────────────────────────────
# REAL DATA SECTION
# ─────────────────────────────────────────────────────────────────────────────
# To use your own data, replace `USE_SYNTHETIC = True` with `False`
# and point DATA_CSV at a CSV file with columns:
#   Teff, logg, feh, age_gyr
# Optional columns (used if present):
#   alpha_fe, parallax_mas, Gmag, bp_rp
#
# Example CSV row:
#   5778, 4.44, 0.00, 4.60, 0.05, 5.0, 10.0, 0.82
# ─────────────────────────────────────────────────────────────────────────────

USE_SYNTHETIC = False       # ← GALAH DR3 is the default now
DATA_CSV      = "stars.csv" # ← only used if USE_SYNTHETIC = False
                             #   and you have your own CSV instead

N_TRAIN = 150_000  # stars for training  (GALAH has ~200k after quality cuts)
N_TEST  =  30_000  # stars for test set

SEED = 42
rng  = np.random.default_rng(SEED)


# ── Step 2: Load / generate data ─────────────────────────────────────────────

if USE_SYNTHETIC:
    print("[1/6] Generating synthetic training data (fallback)...")
    X_all, y_all = make_synthetic_training_data(n=N_TRAIN + N_TEST, seed=SEED)
    idx     = rng.permutation(len(X_all))
    n_tr    = min(N_TRAIN, len(idx) - 50)
    X_train = X_all[idx[:n_tr]]
    y_train = y_all[idx[:n_tr]]
    X_test  = X_all[idx[n_tr:]]
    y_test  = y_all[idx[n_tr:]]

elif DATA_CSV != "stars.csv" and os.path.exists(DATA_CSV):
    print(f"[1/6] Loading custom data from {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    required = ["Teff", "logg", "feh", "age_gyr"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    from spectroage import engineer_features
    X_all = engineer_features(
        Teff      = df["Teff"].values,
        logg      = df["logg"].values,
        feh       = df["feh"].values,
        alpha_fe  = df["alpha_fe"].values    if "alpha_fe"    in df else None,
        parallax  = df["parallax_mas"].values if "parallax_mas" in df else None,
        Gmag      = df["Gmag"].values         if "Gmag"        in df else None,
        bp_rp     = df["bp_rp"].values        if "bp_rp"       in df else None,
    )
    y_all = df["age_gyr"].values
    idx     = rng.permutation(len(X_all))
    n_tr    = min(N_TRAIN, len(idx) - 50)
    X_train = X_all[idx[:n_tr]]
    y_train = y_all[idx[:n_tr]]
    X_test  = X_all[idx[n_tr:]]
    y_test  = y_all[idx[n_tr:]]

else:
    print("[1/6] Downloading GALAH DR3 (real stellar spectra, ~500 MB)...")
    print("      Ages labeled by BSTEP — Bayesian isochrone pipeline")
    print("      This is your Phase 1 benchmark dataset.\n")
    from load_galah import load_galah_dr3
    X_train, X_test, y_train, y_test, df_test = load_galah_dr3(
        max_stars = N_TRAIN + N_TEST,
        test_frac = N_TEST / (N_TRAIN + N_TEST),
        seed      = SEED,
        verbose   = True,
    )

print(f"  Train: {len(X_train)} stars | Test: {len(X_test)} stars")


# ── Step 3: Train ─────────────────────────────────────────────────────────────

print("\n[2/6] Training SpectroAge ensemble...")
sa = SpectroAge(n_ensemble=10, mc_passes=50)
sa.train(X_train, y_train, verbose=True)


# ── Step 4: Predict & evaluate ────────────────────────────────────────────────

print("\n[3/6] Evaluating on held-out test set...")
import time
t0 = time.perf_counter()
y_pred, sigmas = sa.predict(X_test)
elapsed_ms = (time.perf_counter() - t0) / len(X_test) * 1000

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r, _ = stats.pearsonr(y_test, y_pred)
bias = float(np.mean(y_pred - y_test))
mape = float(np.mean(np.abs((y_pred - y_test) / y_test)) * 100)

print(f"\n  {'Metric':<20} {'Value'}")
print(f"  {'─'*35}")
print(f"  {'MAE':<20} {mae:.3f} Gyr")
print(f"  {'RMSE':<20} {rmse:.3f} Gyr")
print(f"  {'Pearson r':<20} {r:.3f}")
print(f"  {'Bias':<20} {bias:+.3f} Gyr")
print(f"  {'MAPE':<20} {mape:.1f}%")
print(f"  {'Speed':<20} {elapsed_ms:.3f} ms/star  ← laptop CPU")
print(f"  {'─'*35}")


# ── Step 5: Calibration ───────────────────────────────────────────────────────

print("\n[4/6] Checking uncertainty calibration...")
cal = sa.calibrate(X_test, y_test)
print(f"  Within 1σ: {cal['within_1sigma_pct']:5.1f}%   ideal: 68.3%")
print(f"  Within 2σ: {cal['within_2sigma_pct']:5.1f}%   ideal: 95.4%")


# ── Step 6: Example predictions ───────────────────────────────────────────────

print("\n[5/6] Example predictions on famous / well-known stars:")
examples = [
    ("Sun",           5778, 4.44,  0.00, 4.60),
    ("Tau Ceti",      5344, 4.49, -0.55, 5.80),
    ("Alpha Cen A",   5790, 4.31,  0.20, 5.30),
    ("61 Cyg A",      4374, 4.63, -0.33, 6.00),
    ("HD 140283",     5777, 3.67, -2.46, 13.5),   # one of oldest known stars
]

print(f"\n  {'Star':<15} {'Pred (Gyr)':>12} {'±σ':>8} {'True (Gyr)':>12}")
print(f"  {'─'*52}")
for name, teff, lg, fe, true_age in examples:
    age, sigma = sa.predict_single(Teff=teff, logg=lg, feh=fe)
    match = "✓" if abs(age - true_age) < 2 * sigma else "~"
    print(f"  {name:<15} {age:>10.2f}  {sigma:>6.2f}  {true_age:>10.2f}  {match}")


# ── Step 7: Save model ────────────────────────────────────────────────────────

print("\n[6/6] Saving trained model...")
sa.save("spectroage_model")
print("  Model saved to spectroage_model/")


# ── Step 8: Generate all plots ────────────────────────────────────────────────

print("\nGenerating publication-quality figures...")
plotter = SpectroAgePlots(output_dir="figures")
plotter.plot_all(sa, X_test, y_test, y_pred, sigmas)

# Display in Colab
try:
    from IPython.display import display, Image
    import os
    fig_files = sorted(Path("figures").glob("*.png"))
    for fp in fig_files:
        print(f"\n── {fp.name} ──")
        display(Image(str(fp)))
except Exception:
    print("(Run in Colab to see inline figures)")

from pathlib import Path

print("\n" + "=" * 60)
print("  SpectroAge — Complete")
print(f"  MAE = {mae:.3f} Gyr  |  {elapsed_ms:.3f} ms/star")
print(f"  Model → spectroage_model/")
print(f"  Plots → figures/")
print("=" * 60)
