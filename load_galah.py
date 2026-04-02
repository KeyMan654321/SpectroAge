"""
SpectroAge — GALAH DR3 Data Loader
====================================
Downloads GALAH DR3 directly from the official VizieR/CDS server,
extracts the columns SpectroAge needs, cleans the data, and returns
a train/test split ready to pass straight into SpectroAge.train().

GALAH DR3 reference
-------------------
Buder et al. 2021, MNRAS 506, 150
Catalog: J/MNRAS/506/150  (VizieR)
Size: ~588,000 stars with spectroscopic parameters + BSTEP ages

How ages are labeled in GALAH DR3
----------------------------------
Ages come from BSTEP (Sharma et al. 2018) — a Bayesian isochrone
pipeline that fits Teff/logg/[Fe/H]/[α/Fe] + photometry against
PARSEC isochrones. This means:
  - SpectroAge trained on GALAH learns to predict BSTEP-equivalent ages
  - Your Phase 1 benchmark IS a direct comparison vs BSTEP/isoclassify
  - Stars with good BSTEP fits have age_bstep and e_age_bstep columns

Usage (Colab)
-------------
    from load_galah import load_galah_dr3
    X_train, X_test, y_train, y_test, df_test = load_galah_dr3()
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

# Direct download URL — GALAH DR3 main catalog on VizieR CDS
GALAH_URL = (
    "https://cdsarc.cds.unistra.fr/viz-bin/nph-catd/"
    "J/MNRAS/506/150/table2.dat.gz"
)

# Fallback: Zenodo DOI direct file link (same content, more stable)
GALAH_ZENODO = (
    "https://zenodo.org/record/4579219/files/GALAH_DR3_main_allstar_v2.fits"
)

# Local cache path
CACHE_DIR  = Path("galah_data")
CACHE_FITS = CACHE_DIR / "GALAH_DR3_main_allstar_v2.fits"
CACHE_CSV  = CACHE_DIR / "galah_dr3_spectroage.csv"

# Quality cuts
MIN_SNR        = 30      # signal-to-noise ratio per pixel
MAX_AGE_ERR    = 3.0     # Gyr — drop stars with huge BSTEP uncertainty
MIN_AGE        = 0.1     # Gyr
MAX_AGE        = 14.0    # Gyr
MAX_TEFF_ERR   = 200.0   # K
MAX_LOGG_ERR   = 0.25    # dex
MAX_FEH_ERR    = 0.15    # dex
FLAG_OK        = 0       # flag_sp == 0 means no known spectral issues


def load_galah_dr3(
    max_stars:   int   = 200_000,
    test_frac:   float = 0.15,
    seed:        int   = 42,
    cache:       bool  = True,
    verbose:     bool  = True,
) -> tuple:
    """
    Download (or load from cache), clean, and split GALAH DR3.

    Returns
    -------
    X_train, X_test : np.ndarray  — feature matrices for SpectroAge
    y_train, y_test : np.ndarray  — BSTEP ages in Gyr (training labels)
    df_test         : pd.DataFrame — full test set with all columns
                       (useful for the benchmark comparison)

    Notes
    -----
    The returned y values ARE the BSTEP ages, so training SpectroAge
    on this data and then comparing predictions to BSTEP on held-out
    stars IS your Phase 1 benchmark.
    """
    if verbose:
        print("=" * 60)
        print("  GALAH DR3 Data Loader")
        print("=" * 60)

    # ── 1. Download / load ────────────────────────────────────────────────────
    df = _load_raw(cache=cache, verbose=verbose)

    # ── 2. Quality cuts ───────────────────────────────────────────────────────
    df = _apply_quality_cuts(df, verbose=verbose)

    # ── 3. Downsample if requested ────────────────────────────────────────────
    if len(df) > max_stars:
        df = df.sample(n=max_stars, random_state=seed).reset_index(drop=True)
        if verbose:
            print(f"  Downsampled to {max_stars:,} stars")

    # ── 4. Build feature matrix ───────────────────────────────────────────────
    from spectroage import engineer_features
    X = engineer_features(
        Teff     = df["teff_bstep"].values,
        logg     = df["logg_bstep"].values,
        feh      = df["fe_h"].values,
        alpha_fe = df["alpha_fe"].values    if "alpha_fe"    in df else None,
        parallax = df["parallax"].values    if "parallax"    in df else None,
        Gmag     = df["Gmag"].values        if "Gmag"        in df else None,
        bp_rp    = df["bp_rp"].values       if "bp_rp"       in df else None,
    )
    y = df["age_bstep"].values

    # ── 5. Train / test split ─────────────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(X))
    n_te  = int(len(X) * test_frac)
    i_te, i_tr = idx[:n_te], idx[n_te:]

    X_train, y_train = X[i_tr], y[i_tr]
    X_test,  y_test  = X[i_te], y[i_te]
    df_test = df.iloc[i_te].reset_index(drop=True)

    if verbose:
        print(f"\n  Final dataset:")
        print(f"    Train : {len(X_train):>7,} stars")
        print(f"    Test  : {len(X_test):>7,} stars")
        print(f"    Age range : {y.min():.2f} – {y.max():.2f} Gyr")
        print(f"    Teff range: {df['teff_bstep'].min():.0f} – "
              f"{df['teff_bstep'].max():.0f} K")
        print()

    return X_train, X_test, y_train, y_test, df_test


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_raw(cache: bool, verbose: bool) -> pd.DataFrame:
    """Download GALAH DR3 FITS or load from CSV cache."""

    # Fast path: pre-cleaned CSV cache exists
    if cache and CACHE_CSV.exists():
        if verbose:
            print(f"\n  Loading from cache: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    # Medium path: FITS already downloaded
    if CACHE_FITS.exists():
        if verbose:
            print(f"\n  Reading FITS file: {CACHE_FITS}")
        return _fits_to_df(CACHE_FITS, cache=cache, verbose=verbose)

    # Download path
    CACHE_DIR.mkdir(exist_ok=True)
    if verbose:
        print(f"\n  Downloading GALAH DR3 (~500 MB) from Zenodo...")
        print(f"  This happens once — cached to {CACHE_FITS}")
        print(f"  URL: {GALAH_ZENODO}\n")

    _download_with_progress(GALAH_ZENODO, CACHE_FITS)
    return _fits_to_df(CACHE_FITS, cache=cache, verbose=verbose)


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream download with a progress bar. Works in Colab."""
    try:
        import requests
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "requests"])
        import requests

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB
    t0 = time.time()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct  = downloaded / total * 100
                    mb   = downloaded / 1e6
                    spd  = downloaded / (time.time() - t0 + 1e-6) / 1e6
                    print(f"  {pct:5.1f}%  {mb:.0f} MB  ({spd:.1f} MB/s)",
                          end="\r")

    print(f"\n  Download complete: {downloaded/1e6:.0f} MB")


def _fits_to_df(
    fits_path: Path,
    cache: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Read FITS, select relevant columns, save CSV cache."""
    try:
        from astropy.table import Table
    except ImportError:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "astropy"])
        from astropy.table import Table

    if verbose:
        print("  Parsing FITS file...")

    tbl = Table.read(str(fits_path))
    df  = tbl.to_pandas()

    # Decode byte-string columns (FITS quirk)
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].str.decode("utf-8").str.strip()
        except Exception:
            pass

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    if verbose:
        print(f"  Raw catalog: {len(df):,} rows, {len(df.columns)} columns")
        _show_available_columns(df)

    # ── Column mapping ────────────────────────────────────────────────────────
    # GALAH DR3 column names (from Buder et al. 2021 Table 2):
    #   teff_bstep, logg_bstep, fe_h, alpha_fe, age_bstep, e_age_bstep
    #   parallax (from Gaia EDR3 cross-match), phot_g_mean_mag, bp_rp
    #   snr_c3_iraf (SNR in green channel), flag_sp (quality flag)
    #   e_teff, e_logg, e_fe_h
    rename = {
        "phot_g_mean_mag": "Gmag",
        "e_age_bstep":     "age_err",
        "e_teff":          "Teff_err",
        "e_logg":          "logg_err",
        "e_fe_h":          "feh_err",
        "snr_c3_iraf":     "snr",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Compute bp_rp if not present but components are
    if "bp_rp" not in df.columns:
        if "phot_bp_mean_mag" in df.columns and "phot_rp_mean_mag" in df.columns:
            df["bp_rp"] = df["phot_bp_mean_mag"] - df["phot_rp_mean_mag"]

    # Keep only the columns we need
    keep = [c for c in [
        "star_id", "sobject_id",
        "teff_bstep", "logg_bstep", "fe_h", "alpha_fe",
        "age_bstep", "age_err",
        "parallax", "parallax_error",
        "Gmag", "bp_rp",
        "Teff_err", "logg_err", "feh_err",
        "snr", "flag_sp", "flag_fe_h",
        "ra_dr2", "dec_dr2",
    ] if c in df.columns]

    df = df[keep].copy()

    if cache:
        df.to_csv(CACHE_CSV, index=False)
        if verbose:
            print(f"  Saved column-filtered CSV to {CACHE_CSV}")

    return df


def _apply_quality_cuts(df: pd.DataFrame, verbose: bool) -> pd.DataFrame:
    """Apply standard GALAH DR3 quality cuts for age estimation."""
    n0 = len(df)

    # Must have age estimate
    df = df.dropna(subset=["age_bstep", "teff_bstep", "logg_bstep", "fe_h"])

    # Physical age range
    df = df[(df["age_bstep"] >= MIN_AGE) & (df["age_bstep"] <= MAX_AGE)]

    # Age uncertainty cut — only keep well-constrained ages
    if "age_err" in df.columns:
        df = df[df["age_err"].fillna(999) <= MAX_AGE_ERR]

    # SNR cut
    if "snr" in df.columns:
        df = df[df["snr"].fillna(0) >= MIN_SNR]

    # Spectral quality flag (0 = good)
    if "flag_sp" in df.columns:
        df = df[df["flag_sp"].fillna(1) == FLAG_OK]

    # Parameter uncertainty cuts
    if "Teff_err" in df.columns:
        df = df[df["Teff_err"].fillna(999) <= MAX_TEFF_ERR]
    if "logg_err" in df.columns:
        df = df[df["logg_err"].fillna(999) <= MAX_LOGG_ERR]
    if "feh_err" in df.columns:
        df = df[df["feh_err"].fillna(999) <= MAX_FEH_ERR]

    df = df.reset_index(drop=True)

    if verbose:
        print(f"\n  Quality cuts: {n0:,} → {len(df):,} stars "
              f"({len(df)/n0*100:.1f}% retained)")
        print(f"  Cuts applied:")
        print(f"    SNR ≥ {MIN_SNR}")
        print(f"    Age uncertainty ≤ {MAX_AGE_ERR} Gyr")
        print(f"    flag_sp == 0 (no spectral issues)")
        print(f"    Age in [{MIN_AGE}, {MAX_AGE}] Gyr")

    return df


def _show_available_columns(df: pd.DataFrame) -> None:
    """Print a summary of key columns found in the catalog."""
    key_cols = [
        "teff_bstep", "logg_bstep", "fe_h", "alpha_fe",
        "age_bstep", "e_age_bstep", "parallax",
        "phot_g_mean_mag", "bp_rp", "snr_c3_iraf", "flag_sp",
    ]
    found   = [c for c in key_cols if c in df.columns]
    missing = [c for c in key_cols if c not in df.columns]
    print(f"  Key columns found   : {found}")
    if missing:
        print(f"  Key columns missing : {missing}")
        print(f"  (will be filled with defaults if optional)")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing GALAH DR3 loader (dry run — no download)...")
    print()
    print("To download and use real data in Colab, run:")
    print()
    print("  from load_galah import load_galah_dr3")
    print("  X_train, X_test, y_train, y_test, df_test = load_galah_dr3()")
    print()
    print("Then train SpectroAge:")
    print()
    print("  from spectroage import SpectroAge")
    print("  sa = SpectroAge()")
    print("  sa.train(X_train, y_train)")
    print("  ages, sigmas = sa.predict(X_test)")
