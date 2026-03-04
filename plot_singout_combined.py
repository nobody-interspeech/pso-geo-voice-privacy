import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from tqdm import tqdm
import csv

mpl.rcParams.update({'font.size': 14})



# BASELINE
results_files_plot2_baseline = (
   '/vol/das-nobackup/users/ethiombiano/lastOne/PSO_results/plot2.L1.2026_01_27-07_28_21_PM.csv',
   '/vol/das-nobackup/users/ethiombiano/lastOne/PSO_results/plot2.L3.2026_01_27-12_43_15_PM.csv',
   '/vol/das-nobackup/users/ethiombiano/lastOne/PSO_results/plot2.L30.2026_01_24-11_27_39_PM.csv',
)

# IGNORANT
results_files_plot2_ignorant = (
   './PSO_results_ATTACKED_BY_IGNORANT_v1_CNIL202310/plot2.L1.anon.2023_11_01-08_47_23_AM.csv',
   './PSO_results_ATTACKED_BY_IGNORANT_v1_CNIL202310/plot2.L3.anon.2023_11_02-10_23_39_AM.csv',
   './PSO_results_ATTACKED_BY_IGNORANT_v1_CNIL202310/plot2.L10.anon.2023_11_02-06_20_09_PM.csv',
   './PSO_results_ATTACKED_BY_IGNORANT_v1_CNIL202310/plot2.L30.anon.2023_11_02-09_34_45_PM.csv'

)

# SEMI-INFORMED
results_files_plot2_semi_informed = (
    './PSO_results_ATTACKED_BY_am_nsf_dense_random__CNIL202310/plot2.L1.anon.2023_11_13-08_31_53_PM.csv',
    './PSO_results_ATTACKED_BY_am_nsf_dense_random__CNIL202310/plot2.L3.anon.2023_11_14-10_22_06_AM.csv',
    './PSO_results_ATTACKED_BY_am_nsf_dense_random__CNIL202310/plot2.L10.anon.2023_11_14-07_49_26_PM.csv',
    './PSO_results_ATTACKED_BY_am_nsf_dense_random__CNIL202310/plot2.L30.anon.2023_11_14-10_34_12_PM.csv'

)

# INFORMED
results_files_plot2_informed = (
    './PSO_results_2025_INFORMED/plot2.L1.anon.2025_02_03-12_06_01_PM.csv',
    './PSO_results_2025_INFORMED/plot2.L3.anon.2025_02_03-12_06_48_PM.csv',
    './PSO_results_2025_INFORMED/plot2.L10.anon.2025_02_03-11_55_33_PM.csv',
    './PSO_results_2025_INFORMED/plot2.L30.anon.2025_02_03-11_55_40_PM.csv'
)

def load_for_plot2(results_files, step=0):
    "L;fold;N;predicate;run;isolation"
    isolation_scores_all = {}
    for results_file in results_files:
        with open(results_file, mode='r', newline='') as f:
            reader = csv.DictReader(f, delimiter=';', 
                                    fieldnames=('L', 'fold', 'N', 'predicate', 'run', 'isolation'))
            for row in tqdm(reader):
                #print(row)
                try:
                    L = int(row['L'])
                    N = int(row['N'])
                    if not step or (N - 20) % step == 0:
                        if L not in isolation_scores_all:
                            isolation_scores_all[L] = {}
                        if N not in isolation_scores_all[L]:
                            isolation_scores_all[L][N] = {}
                        isolation_scores_all[L][N][(row['fold'], row['predicate'], row['run'])] = float(row['isolation'])
                except ValueError:
                    pass
            
    return {L: {N: np.mean(list(isolation_scores_all[L][N].values()) )
                            for N in isolation_scores_all[L]} for L in isolation_scores_all}
  

# Load original data
orig_plot2_data = load_for_plot2(results_files_plot2_baseline)



# Allowed conversation lengths (as strings)
ALLOWED_CLEN = ["1", "3", "30"]

# Create a figure with one subplot per conversation length L
fig, axs = plt.subplots(1, len(ALLOWED_CLEN), figsize=(18, 6), sharex=True, sharey=True)

# Global list to store CSV rows; header columns: attacker, L, enrollment speakers, curve type, mean, std, chance
csv_rows = []
csv_rows.append(["attacker", "L", "enrollment_speakers", "curve_type", "singling_out_mean", "singling_out_std", "chance"])

# Function to export a single curve's data into csv_rows
def export_curve(attacker_label, L, x_vals, means, stds, curve_type):
    for x, m, s in zip(x_vals, means, stds):
        csv_rows.append([attacker_label, L, x, curve_type, m, s, 0.37])


def plot_for_length(ax, conv_length):
    orig_data = orig_plot2_data[int(conv_length)]
    data_key = "plot2"

    ax.axhline(y=0.37, label='trivial', color="gray", linestyle="-.")
    
    # Plot original (non-anonymized) curve
    x_vals = sorted([int(x) for x in orig_data.keys()])
    orig_means = [np.mean(orig_data[x]) for x in x_vals]
    orig_stds = [np.std(orig_data[x]) for x in x_vals]
    ax.errorbar(x_vals, orig_means, yerr=orig_stds, color="black", linestyle="-", marker='o', label="Baseline")

    export_curve("baseline", conv_length, x_vals, orig_means, orig_stds, "orig")
    
    ax.set_xscale('log')
    ax.set_xlabel("Number of speakers in the dataset")
    ax.set_title(f"Conversation length L = {conv_length}")
    ax.set_ylim(0, 1)
    ax.set_xlim(10, 25000)

# Plot subfigures for each conversation length
for idx, conv_length in enumerate(ALLOWED_CLEN):
    plot_for_length(axs[idx], conv_length)

axs[0].set_ylabel("Singling Out")

# Create a single combined legend outside the subplots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("combined_singling_out_baseline5.pdf")

# Export the collected data to a CSV file
#with open("singling_out_results.csv", "w", newline="") as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerows(csv_rows)