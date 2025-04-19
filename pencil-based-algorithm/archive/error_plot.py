import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pencil import pencil_decompose, pencil_recompose
import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose

def plot_tensor_reconstruction_error():
    alphas = [i for i in range(2,100)]

    pencil_errors = []
    bkw_errors = []

    max_err = 0
    max_err_alpha = None    

    for a in alphas:
        c=1
        f1 = [[c,0], [0, c]]
        f2 = [[c,a], [a, c]]
        tensor = np.stack([f1, f2], axis=2).astype(complex)

        pencil_tensor = pencil_recompose(pencil_decompose(tensor))
        factors,factor_matrices = bkw_decompose(tensor)
        bkw_tensor = bkw_recompose(factors,factor_matrices)

        pencil_err = np.abs(tensor - pencil_tensor).mean()
        bkw_err = np.abs(tensor - bkw_tensor).mean()

        if pencil_err > max_err:
            max_err = pencil_err
            max_err_alpha = a

        pencil_errors.append(pencil_err)
        bkw_errors.append(bkw_err)

    order = int(np.floor(math.log10(max_err)))
    rounded = round(max_err / 10**order, 2)
    print(f"Max Pencil Error: error {rounded}, power of 10 : {order} in alpha : {max_err_alpha} or log : {math.log(max_err_alpha,1.5**-1)}")

    """filtered = [(a, e) for a, e in zip(alphas, pencil_errors) if e > 1e-8]
    log_alphas = np.log([a for a, e in filtered])
    log_errors = np.log([e for a, e in filtered])

    coeffs = np.polyfit(log_alphas, log_errors, 1)
    slope, intercept = coeffs
    fitted = np.exp(intercept) * np.array(alphas) ** slope
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))

    sns.lineplot(x=alphas, y=pencil_errors, marker='o', label="Pencil")
    sns.lineplot(x=alphas, y=bkw_errors, marker='s', label="BKW")
    plt.plot(alphas, linestyle='--', color='gray', label="Pencil Error (Quadratic Fit)")


    plt.xscale('log')
    plt.gca()  
    plt.xlabel("alpha")
    plt.ylabel("Mean Absolute Reconstruction Error")
    plt.title("Reconstruction Error vs. Alpha")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_tensor_reconstruction_error()
