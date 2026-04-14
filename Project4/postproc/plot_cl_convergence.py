#!/usr/bin/env python3
"""
Parse solver stderr logs and plot Cl vs number of elements for:
  - Adjoint-adapted sequence  (--adjoint-adapt run on base.gri)
  - Uniform refinement sequence (separate steady runs on 2k/8k/32k/128k grids)

Usage:
    python3 postproc/plot_cl_convergence.py \
        --adapt  data_steady/adapt_run.log \
        --uniform data_steady/uniform_run.log \
        --out    output_final/cl_convergence.png
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── parsers ──────────────────────────────────────────────────────────────────

def parse_adaptation_log(logfile):
    """
    Extract (n_elements, Cl) pairs from an adjoint-adaptation run log.
    Looks for pairs of lines:
        Mesh: <N> elements
        Cl = <value>  (<N> elements)
    The more reliable marker is the "Cl = ... elements" line.
    """
    entries = []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'Cl\s*=\s*([-+eE\d.]+)\s+\(\s*(\d+)\s+elements?\)', line)
            if m:
                cl  = float(m.group(1))
                nel = int(m.group(2))
                entries.append((nel, cl))
    return entries   # list of (n_elements, Cl)


def parse_uniform_log(logfile):
    """
    Extract (n_elements, Cl) pairs from multiple single-mesh steady runs
    concatenated into one log file.
    Same "Cl = <value>  (<N> elements)" marker.
    """
    return parse_adaptation_log(logfile)   # same format


def parse_sensitivity_block(logfile):
    """Extract the adjoint sensitivity validation line if present."""
    adj_val = None
    fd_val  = None
    with open(logfile) as f:
        for line in f:
            m = re.search(r'dCl/dalpha \(adjoint\)\s*=\s*([-+eE\d.]+)', line)
            if m:
                adj_val = float(m.group(1))
            m = re.search(r'dCl/dalpha \(FD\)\s*=\s*([-+eE\d.]+)', line)
            if m:
                fd_val = float(m.group(1))
    return adj_val, fd_val


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_cl_convergence(adapt_entries, uniform_entries, reference_Cl, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))

    if adapt_entries:
        ne_a, cl_a = zip(*sorted(adapt_entries))
        ax.semilogx(ne_a, cl_a, 'b-o', label='Adjoint-adapted', markersize=6)

    if uniform_entries:
        ne_u, cl_u = zip(*sorted(uniform_entries))
        ax.semilogx(ne_u, cl_u, 'r-s', label='Uniform refinement', markersize=6)

    if reference_Cl is not None:
        ax.axhline(reference_Cl, color='k', linestyle='--', linewidth=1,
                   label=f'Reference $C_l$ = {reference_Cl:.4f}')

    ax.set_xlabel('Number of elements')
    ax.set_ylabel(r'$C_l$')
    ax.set_title(r'$C_l$ convergence: adjoint-adapted vs. uniform refinement')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_effectivity(adapt_log, out_path):
    """
    Plot estimated |delta_Cl| (from error indicators) vs actual |delta_Cl|
    (difference in Cl between successive cycles) to assess effectivity.
    """
    entries = parse_adaptation_log(adapt_log)
    if len(entries) < 2:
        print("Not enough adaptation data for effectivity plot.")
        return

    estimated = []
    actual    = []
    nel_vals  = []

    # Parse estimated error from log
    # Matches: "Error indicators: sum |eps_e| = 0.0086 (estimated |delta_Cl|)"
    estimates_raw = []
    with open(adapt_log) as f:
        for line in f:
            m = re.search(r'Error indicators.*sum.*\|eps_e\|\s*=\s*([-+eE\d.]+)', line)
            if m:
                estimates_raw.append(float(m.group(1)))

    n = min(len(entries) - 1, len(estimates_raw))
    for i in range(n):
        nel_vals.append(entries[i][0])
        estimated.append(abs(estimates_raw[i]))
        actual.append(abs(entries[i+1][1] - entries[i][1]))

    if not estimated:
        print("No effectivity data found.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(estimated))
    ax.semilogy(x, estimated, 'b-o', label='Estimated $|\\Delta C_l|$')
    ax.semilogy(x, actual,    'r-s', label='Actual $|\\Delta C_l|$')
    ax.set_xlabel('Adaptation cycle')
    ax.set_ylabel(r'$|\Delta C_l|$')
    ax.set_title('Error estimator effectivity')
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--adapt',   default='data_steady/adapt_run.log',
                   help='Log file from adjoint-adaptation run')
    p.add_argument('--uniform', default='data_steady/uniform_run.log',
                   help='Log file from uniform-refinement runs (all concatenated)')
    p.add_argument('--ref-log', default='data_steady/ref_run.log',
                   help='Log file from high-resolution reference run')
    p.add_argument('--out',     default='output_final/cl_convergence.png',
                   help='Output figure path')
    p.add_argument('--effectivity-out', default='output_final/cl_effectivity.png',
                   help='Effectivity plot output path')
    args = p.parse_args()

    adapt_entries   = []
    uniform_entries = []
    reference_Cl    = None

    if Path(args.adapt).exists():
        adapt_entries = parse_adaptation_log(args.adapt)
        print(f"Adjoint-adapt: {len(adapt_entries)} cycle(s) found")
        for ne, cl in adapt_entries:
            print(f"  {ne:6d} elements  Cl = {cl:.6f}")

        adj_sens, fd_sens = parse_sensitivity_block(args.adapt)
        if adj_sens is not None:
            print(f"\nSensitivity validation (cycle 0):")
            print(f"  dCl/dalpha (adjoint) = {adj_sens:.6f} rad^-1")
            if fd_sens is not None:
                print(f"  dCl/dalpha (FD)      = {fd_sens:.6f} rad^-1")
                print(f"  Relative error       = {abs(adj_sens - fd_sens)/abs(fd_sens):.2e}")
    else:
        print(f"Warning: adapt log not found: {args.adapt}")

    if Path(args.uniform).exists():
        uniform_entries = parse_uniform_log(args.uniform)
        print(f"\nUniform: {len(uniform_entries)} mesh(es) found")
        for ne, cl in uniform_entries:
            print(f"  {ne:6d} elements  Cl = {cl:.6f}")
    else:
        print(f"Warning: uniform log not found: {args.uniform}")

    if Path(args.ref_log).exists():
        ref_entries = parse_uniform_log(args.ref_log)
        if ref_entries:
            # Use highest-element-count result as reference
            reference_Cl = max(ref_entries, key=lambda x: x[0])[1]
            print(f"\nReference Cl (finest mesh) = {reference_Cl:.6f}")

    plot_cl_convergence(adapt_entries, uniform_entries, reference_Cl, args.out)

    if Path(args.adapt).exists():
        plot_effectivity(args.adapt, args.effectivity_out)


if __name__ == '__main__':
    main()
