#!/bin/bash
# ==============================================================
# run_feff8_all.sh
# Run FEFF8 (Windows executable) via Wine on macOS
# for all uranyl FEFF input files
#
# Usage:
#   1. Place feff8.exe in the same directory as this script
#      (or set FEFF8_EXE below to the full path)
#   2. Place all feff_*.inp files in the same directory
#   3. chmod +x run_feff8_all.sh
#   4. ./run_feff8_all.sh
#
# Prerequisites:
#   brew install wine-stable
#
# Output:
#   Each calculation runs in its own subdirectory.
#   Final xmu.dat files are collected into ./results/
# ==============================================================

set -e

# --- Configuration ---
FEFF8_EXE="feff84_87.exe"           # Path to FEFF8 Windows executable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"

# Check for Wine
if ! command -v wine &> /dev/null; then
    echo "ERROR: Wine not found. Install with: brew install wine-stable"
    exit 1
fi

# Check for FEFF8 executable
if [ ! -f "${SCRIPT_DIR}/${FEFF8_EXE}" ]; then
    echo "ERROR: ${FEFF8_EXE} not found in ${SCRIPT_DIR}"
    echo "Place the FEFF8 Windows executable in this directory."
    exit 1
fi

# Find all FEFF input files
INP_FILES=(${SCRIPT_DIR}/feff_*.inp)
if [ ${#INP_FILES[@]} -eq 0 ]; then
    echo "ERROR: No feff_*.inp files found in ${SCRIPT_DIR}"
    exit 1
fi

echo "Found ${#INP_FILES[@]} FEFF input files:"
for f in "${INP_FILES[@]}"; do
    echo "  $(basename $f)"
done
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# --- Run each calculation ---
for INP_FILE in "${INP_FILES[@]}"; do
    BASENAME=$(basename "${INP_FILE}" .inp)
    RUNDIR="${SCRIPT_DIR}/${BASENAME}"

    echo "============================================================"
    echo "Running: ${BASENAME}"
    echo "============================================================"

    # Create run directory and copy input
    mkdir -p "${RUNDIR}"
    cp "${INP_FILE}" "${RUNDIR}/feff.inp"
    cp "${SCRIPT_DIR}/${FEFF8_EXE}" "${RUNDIR}/"

    # Run FEFF8
    cd "${RUNDIR}"
    echo "  Working directory: ${RUNDIR}"
    echo "  Starting FEFF8..."

    wine "${FEFF8_EXE}" > feff_stdout.log 2> feff_stderr.log
    EXIT_CODE=$?

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "  WARNING: FEFF8 exited with code ${EXIT_CODE}"
        echo "  Check ${RUNDIR}/feff_stdout.log for details"
    fi

    # Check for output
    if [ -f "xmu.dat" ]; then
        # Copy xmu.dat to results with descriptive name
        cp "xmu.dat" "${RESULTS_DIR}/${BASENAME}_xmu.dat"
        echo "  SUCCESS: xmu.dat -> results/${BASENAME}_xmu.dat"
    else
        echo "  WARNING: xmu.dat not produced"
        echo "  Check feff_stdout.log and feff_stderr.log"
    fi

    # Also grab chi.dat if produced (useful for comparison)
    if [ -f "chi.dat" ]; then
        cp "chi.dat" "${RESULTS_DIR}/${BASENAME}_chi.dat"
    fi

    # Clean up the copied executable to save space
    rm -f "${RUNDIR}/${FEFF8_EXE}"

    echo ""
done

cd "${SCRIPT_DIR}"

# --- Summary ---
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo ""
echo "Results in: ${RESULTS_DIR}/"
echo ""
ls -la "${RESULTS_DIR}"/*.dat 2>/dev/null || echo "  (no output files)"
echo ""
echo "To plot in Python:"
echo "  import numpy as np"
echo "  import matplotlib.pyplot as plt"
echo ""
echo "  files = {"
for INP_FILE in "${INP_FILES[@]}"; do
    BASENAME=$(basename "${INP_FILE}" .inp)
    LABEL=$(echo "${BASENAME}" | sed 's/feff_//' | sed 's/_/ /g')
    echo "      '${LABEL}': 'results/${BASENAME}_xmu.dat',"
done
echo "  }"
echo ""
echo "  fig, ax = plt.subplots(figsize=(8, 6))"
echo "  for label, path in files.items():"
echo "      data = np.loadtxt(path)"
echo "      ax.plot(data[:, 0], data[:, 3], label=label)  # col 0=E, col 3=mu"
echo "  ax.set_xlabel('Energy (eV)')"
echo "  ax.set_ylabel('Normalized absorption')"
echo "  ax.legend()"
echo "  plt.savefig('results/HERFD_XANES_comparison.png', dpi=300)"
echo "  plt.show()"
