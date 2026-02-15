#!/usr/bin/env python3
"""
lcf_carbonate_validation.py

Linear combination fitting of FEFF-simulated Ca-uranyl-carbonate spectra
against experimental HERFD-XANES of known mixtures.

This is the key validation test for Paper 1:
  - Simulate mono-Ca and di-Ca spectra with FEFF
  - Apply single E0 shift calibrated from the no-Ca experimental standard
  - LCF the simulated spectra against 70/30 and 30/70 experimental mixtures
  - If recovered ratios match known ratios (±5-10%), models are validated

Usage:
    1. Place experimental data files in ./data/ directory
       (2-column format: energy_eV  normalized_mu)
    2. Place FEFF xmu.dat files in ./results/ directory
    3. Edit file paths below
    4. python3 lcf_carbonate_validation.py
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path


# ==============================================================
# CONFIGURATION - Edit these paths
# ==============================================================

# FEFF simulated spectra (from run_feff8_all.sh output)
FEFF_MONO_CA = "results/feff_CaUO2CO3_solution_xmu.dat"
FEFF_DI_CA = "results/feff_Ca2UO2CO3_solution_xmu.dat"
FEFF_NO_CA = "results/feff_UO2CO3_noCa_xmu.dat"

# Experimental HERFD-XANES spectra
# Format: 2-column ASCII, energy (eV) and normalized mu
# UPDATE THESE PATHS to your actual data files
EXP_NO_CA = "data/exp_UO2CO3_noCa.dat"
EXP_MIX_70_30 = "data/exp_70mono_30di.dat"
EXP_MIX_30_70 = "data/exp_30mono_70di.dat"

# LCF energy range (relative to E0, in eV)
# Focus on white line + post-edge where models differ most
LCF_E_MIN = -10  # eV below edge
LCF_E_MAX = 50   # eV above edge

# E0 shift search range
E0_SHIFT_RANGE = (-15, 15)  # eV


# ==============================================================
# FUNCTIONS
# ==============================================================

def load_feff_xmu(filepath):
    """Load FEFF xmu.dat. Returns (energy_rel_to_E0, mu)."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 2]


def load_experimental(filepath):
    """Load experimental spectrum. Returns (energy_eV, mu)."""
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1]


def normalize(energy, mu, pre_range=(-30, -10), post_range=(50, 80)):
    """Edge-step normalize a spectrum."""
    pre_mask = (energy >= pre_range[0]) & (energy <= pre_range[1])
    post_mask = (energy >= post_range[0]) & (energy <= post_range[1])

    if np.sum(pre_mask) < 2 or np.sum(post_mask) < 2:
        # Fallback: simple max normalization
        mu_shifted = mu - np.min(mu)
        return mu_shifted / np.max(mu_shifted)

    pre_val = np.mean(mu[pre_mask])
    post_val = np.mean(mu[post_mask])
    return (mu - pre_val) / (post_val - pre_val)


def interpolate_to_grid(energy, mu, grid):
    """Interpolate spectrum onto common energy grid."""
    f = interp1d(energy, mu, kind='cubic', bounds_error=False, fill_value=0)
    return f(grid)


def calibrate_e0(feff_e, feff_mu, exp_e, exp_mu, e_range=(-10, 50)):
    """
    Find E0 shift that best aligns FEFF spectrum to experimental.
    Shifts the FEFF energy axis: E_aligned = E_feff + E0_shift + E_edge
    """
    # Create common grid from experimental data within range
    exp_mask = (exp_e >= exp_e.min() + 20) & (exp_e <= exp_e.max() - 20)
    grid = exp_e[exp_mask]

    exp_on_grid = interpolate_to_grid(exp_e, exp_mu, grid)

    def residual(params):
        e0_shift = params[0]
        feff_shifted = interpolate_to_grid(feff_e + e0_shift, feff_mu, grid)
        # Scale factor to match overall intensity
        scale = np.dot(exp_on_grid, feff_shifted) / np.dot(feff_shifted, feff_shifted)
        diff = exp_on_grid - scale * feff_shifted
        # Focus on the edge region
        fit_mask = (grid >= e0_shift + e_range[0]) & (grid <= e0_shift + e_range[1])
        if np.sum(fit_mask) < 5:
            return 1e10
        return np.sum(diff[fit_mask] ** 2)

    result = minimize(residual, x0=[0], bounds=[E0_SHIFT_RANGE], method='L-BFGS-B')
    return result.x[0]


def lcf_fit(spec1_e, spec1_mu, spec2_e, spec2_mu, exp_e, exp_mu,
            e_range=(LCF_E_MIN, LCF_E_MAX), constrained=True):
    """
    Linear combination fit: exp = f1*spec1 + f2*spec2

    If constrained: f1 + f2 = 1, 0 <= f1,f2 <= 1
    Returns: f1, f2, r_factor, fitted_spectrum
    """
    # Common energy grid
    e_min = max(spec1_e.min(), spec2_e.min(), exp_e.min(), e_range[0])
    e_max = min(spec1_e.max(), spec2_e.max(), exp_e.max(), e_range[1])
    grid = np.linspace(e_min, e_max, 500)

    s1 = interpolate_to_grid(spec1_e, spec1_mu, grid)
    s2 = interpolate_to_grid(spec2_e, spec2_mu, grid)
    exp = interpolate_to_grid(exp_e, exp_mu, grid)

    if constrained:
        # f2 = 1 - f1, minimize over f1 only
        def residual(f1):
            f1 = f1[0]
            f2 = 1 - f1
            fit = f1 * s1 + f2 * s2
            return np.sum((exp - fit) ** 2)

        result = minimize(residual, x0=[0.5], bounds=[(0, 1)], method='L-BFGS-B')
        f1 = result.x[0]
        f2 = 1 - f1
    else:
        # Unconstrained least squares
        A = np.column_stack([s1, s2])
        coeffs, _, _, _ = np.linalg.lstsq(A, exp, rcond=None)
        f1, f2 = coeffs

    fitted = f1 * s1 + f2 * s2
    r_factor = np.sum((exp - fitted) ** 2) / np.sum(exp ** 2)

    return f1, f2, r_factor, grid, fitted, exp


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("=" * 60)
    print("LCF VALIDATION: Ca-Uranyl-Carbonate HERFD-XANES")
    print("=" * 60)

    # --- Step 1: Load FEFF spectra ---
    print("\nLoading FEFF simulated spectra...")
    feff_files = {
        "no-Ca": FEFF_NO_CA,
        "mono-Ca": FEFF_MONO_CA,
        "di-Ca": FEFF_DI_CA,
    }

    feff_spectra = {}
    for label, fpath in feff_files.items():
        if Path(fpath).exists():
            e, mu = load_feff_xmu(fpath)
            mu_norm = normalize(e, mu)
            feff_spectra[label] = (e, mu_norm)
            print(f"  Loaded {label}: {len(e)} points, E range [{e.min():.1f}, {e.max():.1f}]")
        else:
            print(f"  WARNING: {fpath} not found")

    # --- Step 2: Load experimental spectra ---
    print("\nLoading experimental spectra...")
    exp_files = {
        "no-Ca (exp)": EXP_NO_CA,
        "70/30 (exp)": EXP_MIX_70_30,
        "30/70 (exp)": EXP_MIX_30_70,
    }

    exp_spectra = {}
    for label, fpath in exp_files.items():
        if Path(fpath).exists():
            e, mu = load_experimental(fpath)
            exp_spectra[label] = (e, mu)
            print(f"  Loaded {label}: {len(e)} points")
        else:
            print(f"  NOT FOUND: {fpath}")
            print(f"  (Create this file with your experimental data)")

    # --- Step 3: Calibrate E0 ---
    if "no-Ca" in feff_spectra and "no-Ca (exp)" in exp_spectra:
        print("\nCalibrating E0 shift using no-Ca standard...")
        e0_shift = calibrate_e0(
            feff_spectra["no-Ca"][0], feff_spectra["no-Ca"][1],
            exp_spectra["no-Ca (exp)"][0], exp_spectra["no-Ca (exp)"][1]
        )
        print(f"  E0 shift = {e0_shift:.2f} eV")
        print(f"  Applying same shift to ALL FEFF spectra")

        # Apply shift
        for label in feff_spectra:
            e, mu = feff_spectra[label]
            feff_spectra[label] = (e + e0_shift, mu)
    else:
        print("\nWARNING: Cannot calibrate E0 (missing no-Ca spectra)")
        print("  Proceeding without E0 correction")
        e0_shift = 0

    # --- Step 4: LCF of known mixtures ---
    if "mono-Ca" in feff_spectra and "di-Ca" in feff_spectra:
        print("\n" + "=" * 60)
        print("LINEAR COMBINATION FITTING")
        print("=" * 60)

        mixtures = {
            "70/30 (exp)": (0.70, 0.30),  # expected fractions
            "30/70 (exp)": (0.30, 0.70),
        }

        fig, axes = plt.subplots(len(mixtures), 1, figsize=(8, 5 * len(mixtures)))
        if len(mixtures) == 1:
            axes = [axes]

        for idx, (mix_label, (f_mono_expected, f_di_expected)) in enumerate(mixtures.items()):
            if mix_label not in exp_spectra:
                print(f"\n  {mix_label}: experimental data not available, skipping")
                continue

            print(f"\n  Fitting: {mix_label}")
            print(f"  Expected: {f_mono_expected:.0%} mono-Ca + {f_di_expected:.0%} di-Ca")

            f1, f2, r_factor, grid, fitted, exp_on_grid = lcf_fit(
                feff_spectra["mono-Ca"][0], feff_spectra["mono-Ca"][1],
                feff_spectra["di-Ca"][0], feff_spectra["di-Ca"][1],
                exp_spectra[mix_label][0], exp_spectra[mix_label][1],
            )

            print(f"  Fitted:   {f1:.1%} mono-Ca + {f2:.1%} di-Ca")
            print(f"  R-factor: {r_factor:.6f}")
            print(f"  Error:    mono-Ca {abs(f1 - f_mono_expected):.1%}, "
                  f"di-Ca {abs(f2 - f_di_expected):.1%}")

            # Assess validation
            mono_error = abs(f1 - f_mono_expected)
            if mono_error <= 0.05:
                verdict = "EXCELLENT (<5%)"
            elif mono_error <= 0.10:
                verdict = "GOOD (<10%)"
            elif mono_error <= 0.15:
                verdict = "ACCEPTABLE (<15%)"
            else:
                verdict = "POOR (>15%) - check models"
            print(f"  Validation: {verdict}")

            # Plot
            ax = axes[idx]
            ax.plot(grid, exp_on_grid, 'k-', lw=2, label=f'Experimental {mix_label}')
            ax.plot(grid, fitted, 'r--', lw=2,
                    label=f'LCF: {f1:.0%} mono + {f2:.0%} di (R={r_factor:.4f})')
            ax.plot(grid, exp_on_grid - fitted + np.min(exp_on_grid) - 0.1,
                    'g-', lw=1, label='Residual')
            ax.set_xlabel('Energy relative to E$_0$ (eV)', fontsize=11)
            ax.set_ylabel('Normalized $\\mu$(E)', fontsize=11)
            ax.set_title(f'{mix_label} — Expected: {f_mono_expected:.0%}/{f_di_expected:.0%}',
                        fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xlim(LCF_E_MIN, LCF_E_MAX)

        fig.tight_layout()
        fig.savefig("results/LCF_validation.png", dpi=300)
        print(f"\nSaved: results/LCF_validation.png")
        plt.show()

    else:
        print("\nCannot perform LCF: need both mono-Ca and di-Ca FEFF spectra")

    # --- Demo mode: if no experimental data, show FEFF-only LCF ---
    if not exp_spectra and len(feff_spectra) >= 2:
        print("\n" + "=" * 60)
        print("DEMO MODE: Synthetic mixture test (no experimental data)")
        print("=" * 60)

        if "mono-Ca" in feff_spectra and "di-Ca" in feff_spectra:
            e_mono, mu_mono = feff_spectra["mono-Ca"]
            e_di, mu_di = feff_spectra["di-Ca"]

            # Create synthetic mixtures
            grid = np.linspace(
                max(e_mono.min(), e_di.min()),
                min(e_mono.max(), e_di.max()),
                500
            )
            s_mono = interpolate_to_grid(e_mono, mu_mono, grid)
            s_di = interpolate_to_grid(e_di, mu_di, grid)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            for ax, (f_true, label) in zip(axes, [(0.70, "70/30"), (0.30, "30/70")]):
                # Generate synthetic mixture with small noise
                synthetic = f_true * s_mono + (1 - f_true) * s_di
                noise = np.random.normal(0, 0.005, len(synthetic))
                synthetic += noise

                # Fit it back
                A = np.column_stack([s_mono, s_di])
                coeffs, _, _, _ = np.linalg.lstsq(A, synthetic, rcond=None)
                fitted = coeffs[0] * s_mono + coeffs[1] * s_di
                r_fac = np.sum((synthetic - fitted)**2) / np.sum(synthetic**2)

                ax.plot(grid, synthetic, 'k-', lw=2, label=f'Synthetic {label}')
                ax.plot(grid, fitted, 'r--', lw=1.5,
                        label=f'LCF: {coeffs[0]:.1%} mono + {coeffs[1]:.1%} di')
                ax.set_xlabel('Energy (eV)', fontsize=11)
                ax.set_ylabel('$\\mu$(E)', fontsize=11)
                ax.set_title(f'True: {f_true:.0%} mono + {1-f_true:.0%} di', fontsize=12)
                ax.legend(fontsize=9)
                ax.set_xlim(LCF_E_MIN, LCF_E_MAX)

                print(f"  {label}: True f_mono={f_true:.2f}, "
                      f"Recovered f_mono={coeffs[0]:.4f}, R={r_fac:.6f}")

            fig.suptitle("Synthetic Mixture Recovery Test", fontsize=13, y=1.02)
            fig.tight_layout()
            fig.savefig("results/LCF_demo_synthetic.png", dpi=300)
            print(f"\n  Saved: results/LCF_demo_synthetic.png")


if __name__ == "__main__":
    main()
