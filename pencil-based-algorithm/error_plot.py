import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pencil import pencil_decompose, pencil_recompose

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose

def plot_tensor_reconstruction_error():
    alphas = [10**-i for i in range(1, 8)]
    pencil_errors = []
    bkw_errors = []

    for a in alphas:
        f1 = [[1, -1], [1, 1]]
        f2 = [[1, -(1 - a)], [(1 - a), 1]]
        tensor = np.stack([f1, f2], axis=2).astype(complex)

        pencil_tensor = pencil_recompose(pencil_decompose(tensor))
        factors,factor_matrices = bkw_decompose(tensor)
        bkw_tensor = bkw_recompose(factors,factor_matrices)

        pencil_err = np.abs(tensor - pencil_tensor).mean()
        bkw_err = np.abs(tensor - bkw_tensor).mean()

        pencil_errors.append(pencil_err)
        bkw_errors.append(bkw_err)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    sns.lineplot(x=alphas, y=pencil_errors, marker='o', label="Pencil")
    #sns.lineplot(x=alphas, y=bkw_errors, marker='s', label="BKW")

    plt.xscale('log')
    plt.gca().invert_xaxis()  # smaller values to the right
    plt.xlabel("alpha (log scale, decreasing →)")
    plt.ylabel("Mean Absolute Reconstruction Error")
    plt.title("Reconstruction Error vs. Alpha")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_tensor_reconstruction_error()
