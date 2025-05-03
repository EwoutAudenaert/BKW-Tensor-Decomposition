import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from exp_library import get_algo_error

# This script plots the distribution of forward decomposition errors for tensors using both the BKW-based and pencil-based algorithms.
# The experiment is done with tensors that have orthogonal factor matrices.
# For BKW, the factors are integer matrices, resulting in tensors with condition number 1.
# size: number of random tensors tested
# n: dimension of the cubic tensors (n x n x n) used in the experiment
# In this figure, size is set to 10^3 and n is set to 3, but n can be changed.

size = 10**3
n=3

results_pencil = []
results_bkw = []
for iter in range(size):
    results_pencil.append(get_algo_error('pencil',n))
    results_bkw.append(get_algo_error('bkw',n))
results_pencil = np.real(results_pencil)
results_bkw = np.real(results_bkw)

mean_pencil = np.mean(results_pencil)
median_pencil = np.median(results_pencil)
mean_bkw = np.mean(results_bkw)
median_bkw = np.median(results_bkw)

sns.set(style="whitegrid", font_scale=1.5)
plt.figure(figsize=(12, 7))

bins = 'auto'

ax = sns.histplot(results_pencil, bins=bins, kde=True, log_scale=(True, False),
                  color="royalblue", edgecolor="white", label="Pencil-based", alpha=0.5)
sns.histplot(results_bkw, bins=bins, kde=True, log_scale=(True, False),
             color="crimson", edgecolor="white", label="BKW", alpha=0.5)

#fixing the meidan block
for container in ax.containers:
    for patch in container.patches:
        bar_left = patch.get_x()
        bar_right = patch.get_x() + patch.get_width()
        if bar_left <= median_pencil <= bar_right:
            patch.set_facecolor('blue')  # darker blue
            patch.set_alpha(1.0)
        if bar_left <= median_bkw <= bar_right:
            patch.set_facecolor('darkred')  # darker red
            patch.set_alpha(1.0)

plt.axvline(mean_pencil, color="navy", linestyle="--", linewidth=2, label=f"Pencil Mean = {mean_pencil:.1e}")
plt.axvline(mean_bkw, color="darkred", linestyle="--", linewidth=2, label=f"BKW Mean = {mean_bkw:.1e}")

plt.title("Log-x distribution of forward decomposition error", fontsize=18, fontweight='regular')
plt.xlabel("Error (log-scale)", fontsize=15)
plt.ylabel("Frequency", fontsize=15)

median_legend_handles = [
    plt.Line2D([0], [0], color='blue', lw=8, label=f"Pencil Median = {median_pencil:.1e}"),
    plt.Line2D([0], [0], color='darkred', lw=8, label=f"BKW Median = {median_bkw:.1e}")
]
plt.legend(handles=ax.get_legend_handles_labels()[0] + median_legend_handles, loc='upper right', frameon=True, fancybox=True)

plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()