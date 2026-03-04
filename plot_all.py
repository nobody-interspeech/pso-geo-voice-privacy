#!/usr/bin/env python3
# coding: utf-8

import os
import re
import csv
import glob
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "legend.fontsize": 22,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
})

BASE_RESULTS_DIR = "/srv/storage/talc3@storage4.nancy/multispeech/calcul/projets/PSO_results"

L_LIST = [1, 3, 30]
GEO_LEVELS = ["household_id", "OA", "LSOA", "MSOA"]  # SIGNAL ONLY is drawn from points below
CHANCE = 0.37

PATTERN = os.path.join(BASE_RESULTS_DIR, "plot2.L{L}.{geo}.2026_02_17*.csv")

# High contrast colors
GEO_COLORS = {
    "household_id": "#1565C0",
    "OA":           "#FF6D00",
    "LSOA":         "#2E7D32",
    "MSOA":         "#7B1FA2",
}

# Your red curve points (source points)
INFORMED_POINTS = {
    1: {
        "x": [30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        "y": [0.37, 0.36, 0.35, 0.35, 0.34, 0.34, 0.35, 0.36, 0.37],
    },
    3: {
        "x": [30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        "y": [0.36, 0.34, 0.33, 0.34, 0.33, 0.34, 0.34, 0.35, 0.35],
    },
    30: {
        "x": [30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        "y": [0.42, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.31],
    },
}

# ✅ Force the red curve to stop exactly where your other curves stop
N_GRID_BY_L = {
    1:  [30, 100, 200, 400, 800, 1600, 3200, 6400, 12800],
    3:  [30, 100, 200, 400, 800, 1600, 3200, 6400, 12800],
    30: [30, 100, 200, 400, 800, 1600, 3200],
}

SIGNAL_ONLY_COLOR = "#D32F2F"  # vivid red

# Hardcoded geo curves (when CSV not available or override needed)
HARDCODED_GEO = {
    3: {
        "household_id": {
            "x": [30, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
            "y": [0.995, 0.993, 0.990, 0.982, 0.972, 0.958, 0.885, 0.865, 0.802],
        }
    }
}


def infer_geo_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"plot2\.L(\d+)\.([A-Za-z0-9_]+)\..*\.csv$", name)
    if not m:
        return None
    return m.group(2)


def collect_files_for(L, geo):
    return sorted(glob.glob(PATTERN.format(L=L, geo=geo)))


def load_many_plot2(files):
    """
    Returns: data[L][geo][N] = list of iso values
    """
    data = {}
    for path in files:
        geo = infer_geo_from_filename(path)
        if geo is None:
            continue

        with open(path, "r", newline="") as f:
            reader = csv.DictReader(
                f,
                delimiter=";",
                fieldnames=("L", "fold", "N", "predicate", "run", "isolation"),
            )
            for row in reader:
                try:
                    L = int(row["L"])
                    N = int(row["N"])
                    iso = float(row["isolation"])
                except Exception:
                    continue
                data.setdefault(L, {}).setdefault(geo, {}).setdefault(N, []).append(iso)
    return data


def summarize(data):
    """
    Returns: summary[L][geo] = (Ns_sorted, means, stds)
    """
    summary = {}
    for L in data:
        summary[L] = {}
        for geo in data[L]:
            Ns = sorted(data[L][geo].keys())
            means = [float(np.mean(data[L][geo][N])) for N in Ns]
            stds  = [float(np.std(data[L][geo][N]))  for N in Ns]
            summary[L][geo] = (Ns, means, stds)
    return summary


def interp_logx(x_src, y_src, x_target):
    """Interpolate in log10(x) (since x-axis is log scale)."""
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_t   = np.asarray(x_target, dtype=float)

    lx_src = np.log10(x_src)
    lx_t   = np.log10(x_t)

    order = np.argsort(lx_src)
    lx_src = lx_src[order]
    y_src  = y_src[order]

    # No extrapolation needed because our target grid is within [30, 12800] or [30, 3200]
    y_t = np.interp(lx_t, lx_src, y_src)
    return y_t.tolist()


def plot_all(summary, out_pdf):
    fig, axs = plt.subplots(1, len(L_LIST), figsize=(20, 7), sharex=True, sharey=True)

    label_to_handle = {}

    for i, L in enumerate(L_LIST):
        ax = axs[i]
        ax.set_title(f"Conversation length L = {L}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of speakers in the dataset")
        ax.set_ylim(0, 1)
        ax.set_xlim(10, 25000)

        # trivial line
        trivial_line = ax.axhline(y=CHANCE, color="gray", linestyle="-.", linewidth=2.2, label="trivial")
        label_to_handle.setdefault("trivial", trivial_line)

        # plot CSV curves
        if L in summary:
            for geo in GEO_LEVELS:
                # Check for hardcoded override first
                if L in HARDCODED_GEO and geo in HARDCODED_GEO[L]:
                    hc = HARDCODED_GEO[L][geo]
                    x_grid = N_GRID_BY_L.get(L, hc["x"])
                    y_interp = interp_logx(hc["x"], hc["y"], x_grid)
                    (line,) = ax.plot(
                        x_grid, y_interp,
                        linestyle="-",
                        marker="o",
                        markersize=6,
                        linewidth=2.2,
                        color=GEO_COLORS.get(geo, "black"),
                        label=geo,
                        zorder=6,
                    )
                    label_to_handle.setdefault(geo, line)
                    continue
                if geo not in summary[L]:
                    continue
                x, y, _ = summary[L][geo]
                (line,) = ax.plot(
                    x, y,
                    linestyle="-",
                    marker="o",
                    markersize=6,
                    linewidth=2.2,
                    color=GEO_COLORS.get(geo, "black"),
                    label=geo,
                    zorder=6,
                )
                label_to_handle.setdefault(geo, line)

        # plot SIGNAL ONLY (red) on the exact grid you gave
        if L in INFORMED_POINTS and L in N_GRID_BY_L:
            x_grid = N_GRID_BY_L[L]
            y_red = interp_logx(INFORMED_POINTS[L]["x"], INFORMED_POINTS[L]["y"], x_grid)

            (signal_only_line,) = ax.plot(
                x_grid, y_red,
                linestyle="-",
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=SIGNAL_ONLY_COLOR,
                label="SIGNAL ONLY",
                zorder=5,
            )
            label_to_handle.setdefault("SIGNAL ONLY", signal_only_line)

    axs[0].set_ylabel("Singling Out")

    # Legend order: trivial + geos + SIGNAL ONLY
    legend_labels = ["trivial"] + [g for g in GEO_LEVELS if g in label_to_handle]
    if "SIGNAL ONLY" in label_to_handle:
        legend_labels.append("SIGNAL ONLY")
    legend_handles = [label_to_handle[lbl] for lbl in legend_labels]

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=6, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_pdf)
    print(f"[OK] saved {out_pdf}")


if __name__ == "__main__":
    all_files = []
    for L in L_LIST:
        for geo in GEO_LEVELS:
            files = collect_files_for(L, geo)
            if not files:
                print(f"[WARN] No files for L={L}, geo={geo}")
            all_files.extend(files)

    print(f"[INFO] total csv files found: {len(all_files)}")
    data = load_many_plot2(all_files)
    summary = summarize(data)

    out_pdf = os.path.join(BASE_RESULTS_DIR, "Figure41.pdf")
    plot_all(summary, out_pdf=out_pdf)
