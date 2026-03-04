#!/usr/bin/env python3
# coding: utf-8

import os
import re
import csv
import glob
import argparse
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


def infer_geo_from_filename(path):
    name = os.path.basename(path)
    m = re.match(r"plot2\.L(\d+)\.([A-Za-z0-9_]+)\..*\.csv$", name)
    if not m:
        return None
    return m.group(2)


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


def plot_all(summary, out_pdf, L_LIST, CHANCE, GEO_LEVELS, GEO_COLORS):
    fig, axs = plt.subplots(1, len(L_LIST), figsize=(20, 7), sharex=True, sharey=True)

    label_to_handle = {}

    for i, L in enumerate(L_LIST):
        ax = axs[i]
        ax.set_title(f"Conversation length L = {L}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of speakers in the dataset")
        ax.set_ylim(0, 1)
        ax.set_xlim(10, 25000)

        trivial_line = ax.axhline(y=CHANCE, color="gray", linestyle="-.", linewidth=2.2, label="trivial")
        label_to_handle.setdefault("trivial", trivial_line)

        if L in summary:
            for geo in GEO_LEVELS:
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

    axs[0].set_ylabel("Singling Out")

    legend_labels = ["trivial"] + [g for g in GEO_LEVELS if g in label_to_handle]
    legend_handles = [label_to_handle[lbl] for lbl in legend_labels]

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=6, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_pdf)
    print(f"[OK] saved {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description='Plot PSO evaluation results with geographic constraints')
    parser.add_argument('--results-dir', required=True, help='Path to results directory containing CSV files')
    args = parser.parse_args()
    
    BASE_RESULTS_DIR = args.results_dir

    L_LIST = [1, 3, 30]
    GEO_LEVELS = ["household_id", "OA", "LSOA", "MSOA"]
    CHANCE = 0.37

    PATTERN = os.path.join(BASE_RESULTS_DIR, "plot2.L{L}.{geo}.*.csv")

    GEO_COLORS = {
        "household_id": "#1565C0",
        "OA":           "#FF6D00",
        "LSOA":         "#2E7D32",
        "MSOA":         "#7B1FA2",
    }

    all_files = []
    for L in L_LIST:
        for geo in GEO_LEVELS:
            files = sorted(glob.glob(PATTERN.format(L=L, geo=geo)))
            if not files:
                print(f"[WARN] No files for L={L}, geo={geo}")
            all_files.extend(files)

    print(f"[INFO] total csv files found: {len(all_files)}")
    data = load_many_plot2(all_files)
    summary = summarize(data)

    out_pdf = os.path.join(BASE_RESULTS_DIR, "Figure41.pdf")
    plot_all(summary, out_pdf=out_pdf, L_LIST=L_LIST, CHANCE=CHANCE, GEO_LEVELS=GEO_LEVELS, 
             GEO_COLORS=GEO_COLORS)


if __name__ == "__main__":
    main()
