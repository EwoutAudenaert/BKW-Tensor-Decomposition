import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
from exp_library import get_algo_error
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This script plots the mean decomposition error for tensors using both the BKW-based and pencil-based algorithms.
# n: maximum dimension for tensors (n x n x n) to plot the decomposition error for
# In the figure, n is set to 15
n=15


dims = range(3, n+1)
errors_pencil = []
errors_bkw = []
for n in dims:
    for _ in range(50):
        errors_pencil.append((n, get_algo_error('pencil', n)))
        errors_bkw.append((n, get_algo_error('bkw', n)))

errors_pencil = np.array(errors_pencil)
errors_bkw = np.array(errors_bkw)

# plot
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

sns.lineplot(x=errors_pencil[:, 0], y=errors_pencil[:, 1], label="pencil", marker="o")
sns.lineplot(x=errors_bkw[:, 0], y=errors_bkw[:, 1], label="bkw", marker="s")

plt.yscale('log')
plt.title("Forward decomposition error vs Dimension")
plt.xlabel("Dimension (n)")
plt.ylabel("Error (log-scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()