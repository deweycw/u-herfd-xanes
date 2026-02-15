#!/usr/bin/env python3
"""
plot_feff_results.py

Plot and compare FEFF8 xmu.dat output files for all uranyl models.
Run after run_feff8_all.sh has completed.

Usage:
    python3 plot_feff_results.py

Expects xmu.dat files in ./results/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

RESULTS_DIR = Path("results")

# Define models and groupings
MODELS = {
    # Carbonate models (from earlier session - include if present)
    "feff_CaUO2CO3_solution_xmu.dat": {
        "label": "Mono-Ca carbonate",
        "group": "carbonate",
        "color": "#1f77b4",
        "ls": "-",
    },
    "feff_Ca2UO2CO3_solution_xmu.dat": {
        "label": "Di-Ca carbonate",
        "group": "carbonate",
        "color": "#ff7f0e",
        "ls": "-",
    },
    "feff_UO2CO3_noCa_xmu.dat": {
        "label": "No-Ca carbonate",
        "group": "carbonate",
        "color": "#2ca02c",
        "ls": "-",
    },
    "feff_liebigite_full_xmu.dat": {
        "label": "Liebigite (solid)",
        "group": "carbonate",
        "color": "#9467bd",
        "ls": "--",
    },
    # Organic models
    "feff_UO2_oxalate_xmu.dat": {
        "label": "Bis-oxalato",
        "group": "organic",
        "color": "#1f77b4",
        "ls": "-",
    },
    "feff_UO2_EDTA_xmu.dat": {
        "label": "EDTA",
        "group": "organic",
        "color": "#ff7f0e",
        "ls": "-",
    },
    "feff_UO2_IP6_mono_xmu.dat": {
        "label": "IP6 monodentate",
        "group": "IP6",
        "color": "#d62728",
        "ls": "-",
    },
    "feff_UO2_IP6_bident_xmu.dat": {
        "label": "IP6 bidentate",
        "group": "IP6",
        "color": "#2ca02c",
        "ls": "-",
    },
    "feff_UO2_IP6_chelate_xmu.dat": {
        "label": "IP6 chelate",
        "group": "IP6",
        "color": "#9467bd",
        "ls": "-",
    },
}


def load_xmu(filepath):
    """Load FEFF xmu.dat file. Returns energy (eV) and mu arrays."""
    data = np.loadtxt(filepath)
    # xmu.dat columns: E(eV)  k  mu  mu0  chi
    # Column 0 = energy relative to edge (E - E0)
    # Column 2 = mu (total absorption)
    return data[:, 1], data[:, 3]


def normalize_spectrum(energy, mu, e_range=(-20, -5)):
    """Simple edge-step normalization."""
    # Find pre-edge region for baseline
    pre_mask = (energy >= e_range[0]) & (energy <= e_range[1])
    if np.sum(pre_mask) > 2:
        pre_avg = np.mean(mu[pre_mask])
    else:
        pre_avg = mu[0]

    # Find white line maximum for normalization
    post_mask = (energy >= 0) & (energy <= 50)
    if np.sum(post_mask) > 0:
        wl_max = np.max(mu[post_mask])
    else:
        wl_max = np.max(mu)

    mu_norm = (mu - pre_avg) / (wl_max - pre_avg)
    return mu_norm


def apply_e0_shift(energy, shift_eV):
    """Apply uniform energy shift (for aligning to experiment)."""
    return energy + shift_eV


def main():
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found. Run run_feff8_all.sh first.")
        sys.exit(1)

    # Find available files
    available = {}
    for fname, info in MODELS.items():
        fpath = RESULTS_DIR / fname
        if fpath.exists():
            available[fname] = info

    if not available:
        print("ERROR: No xmu.dat files found in results/")
        print("Available files:")
        for f in RESULTS_DIR.glob("*.dat"):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Found {len(available)} spectra to plot")

    # ---- Figure 1: All spectra overview ----
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    offset = 0
    for fname, info in available.items():
        energy, mu = load_xmu(RESULTS_DIR / fname)
        mu_norm = normalize_spectrum(energy, mu)
        ax1.plot(
            energy,
            mu_norm + offset,
            color=info["color"],
            ls=info["ls"],
            lw=1.5,
            label=info["label"],
        )
        offset += 0.3  # stack spectra

    ax1.set_xlabel("Energy relative to E$_0$ (eV)", fontsize=12)
    ax1.set_ylabel("Normalized $\\mu$(E) + offset", fontsize=12)
    ax1.set_xlim(-20, 80)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_title("FEFF8 Simulated U L$_3$-edge HERFD-XANES", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(RESULTS_DIR / "all_spectra_stacked.png", dpi=300)
    print(f"Saved: {RESULTS_DIR}/all_spectra_stacked.png")

    # ---- Figure 2: IP6 model comparison (key diagnostic plot) ----
    ip6_files = {k: v for k, v in available.items() if v["group"] == "IP6"}
    if len(ip6_files) >= 2:
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(8, 8), height_ratios=[3, 1])

        spectra = {}
        for fname, info in ip6_files.items():
            energy, mu = load_xmu(RESULTS_DIR / fname)
            mu_norm = normalize_spectrum(energy, mu)
            spectra[info["label"]] = (energy, mu_norm)
            ax2a.plot(
                energy, mu_norm, color=info["color"], lw=2, label=info["label"]
            )

        ax2a.set_xlim(-15, 60)
        ax2a.set_ylabel("Normalized $\\mu$(E)", fontsize=12)
        ax2a.legend(fontsize=10)
        ax2a.set_title(
            "IP6 Coordination Mode Discrimination", fontsize=13
        )
        ax2a.axvline(0, color="gray", ls=":", alpha=0.5)

        # Difference spectra (relative to monodentate)
        labels = list(spectra.keys())
        if "IP6 monodentate" in spectra:
            ref_e, ref_mu = spectra["IP6 monodentate"]
            for label in labels:
                if label == "IP6 monodentate":
                    continue
                e, mu = spectra[label]
                # Interpolate to common grid
                mu_interp = np.interp(ref_e, e, mu)
                diff = mu_interp - ref_mu
                info = [v for v in ip6_files.values() if v["label"] == label][0]
                ax2b.plot(ref_e, diff, color=info["color"], lw=1.5, label=f"{label} − monodentate")

            ax2b.axhline(0, color="gray", ls=":", alpha=0.5)
            ax2b.set_xlim(-15, 60)
            ax2b.set_xlabel("Energy relative to E$_0$ (eV)", fontsize=12)
            ax2b.set_ylabel("$\\Delta\\mu$(E)", fontsize=12)
            ax2b.legend(fontsize=9)

        fig2.tight_layout()
        fig2.savefig(RESULTS_DIR / "IP6_model_comparison.png", dpi=300)
        print(f"Saved: {RESULTS_DIR}/IP6_model_comparison.png")

    # ---- Figure 3: Carbonate comparison (if available) ----
    carb_files = {k: v for k, v in available.items() if v["group"] == "carbonate"}
    if len(carb_files) >= 2:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        for fname, info in carb_files.items():
            energy, mu = load_xmu(RESULTS_DIR / fname)
            mu_norm = normalize_spectrum(energy, mu)
            ax3.plot(energy, mu_norm, color=info["color"], ls=info["ls"], lw=1.5, label=info["label"])

        ax3.set_xlim(-15, 60)
        ax3.set_xlabel("Energy relative to E$_0$ (eV)", fontsize=12)
        ax3.set_ylabel("Normalized $\\mu$(E)", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.set_title("Ca-Uranyl-Carbonate HERFD-XANES", fontsize=13)
        fig3.tight_layout()
        fig3.savefig(RESULTS_DIR / "carbonate_comparison.png", dpi=300)
        print(f"Saved: {RESULTS_DIR}/carbonate_comparison.png")

    # ---- Figure 4: Organic ligand comparison ----
    org_files = {k: v for k, v in available.items() if v["group"] == "organic"}
    if len(org_files) >= 2:
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        for fname, info in org_files.items():
            energy, mu = load_xmu(RESULTS_DIR / fname)
            mu_norm = normalize_spectrum(energy, mu)
            ax4.plot(energy, mu_norm, color=info["color"], ls=info["ls"], lw=1.5, label=info["label"])

        ax4.set_xlim(-15, 60)
        ax4.set_xlabel("Energy relative to E$_0$ (eV)", fontsize=12)
        ax4.set_ylabel("Normalized $\\mu$(E)", fontsize=12)
        ax4.legend(fontsize=10)
        ax4.set_title("Uranyl-Organic Ligand HERFD-XANES", fontsize=13)
        fig4.tight_layout()
        fig4.savefig(RESULTS_DIR / "organic_comparison.png", dpi=300)
        print(f"Saved: {RESULTS_DIR}/organic_comparison.png")

    plt.show()
    print("\nDone.")


if __name__ == "__main__":
    main()
