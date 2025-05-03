import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.optimize import curve_fit
import sys
import os
from exp_library import random_orthogonal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pencil-based-algorithm')))
from pencil import pencil_decompose, pencil_recompose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose

# This script plots the runtime of the BKW-based and pencil-based decomposition algorithms as a function of tensor dimension.
# start: minimum dimension n of the cubic tensors (n x n x n) tested
# end: maximum dimension n of the cubic tensors (n x n x n) tested
# In this figure, start is set to 3 and end is set to 20.
start = 3
end = 20


def measure_time(algo, n, repetitions=20):
    times = []
    for _ in range(repetitions):
        factor_matrices = [random_orthogonal(n) for _ in range(3)]

        if algo == 'bkw':
            tensor = bkw_recompose([i for i in range(1, n+1)], factor_matrices)
            start = time.time()
            _ = bkw_decompose(tensor)
            end = time.time()

        elif algo == 'pencil':
            tensor = pencil_recompose(factor_matrices)
            start = time.time()
            _ = pencil_decompose(tensor)
            end = time.time()

        times.append(end - start)
    return np.mean(times)

end=end+1
dims = range(start, end + 1)
dims_array = np.array(list(dims))

times_pencil = []
times_bkw = []

for n in dims:
    times_bkw.append(measure_time('bkw', n))
    times_pencil.append(measure_time('pencil', n))

def n4(x, a):
    return a * x**4

def n7(x, a):
    return a * x**7

params_pencil, _ = curve_fit(n4, dims_array, times_pencil)
params_bkw, _ = curve_fit(n7, dims_array, times_bkw)

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

plt.plot(dims_array, times_pencil, 'o-', label='Pencil-based', color='royalblue')
plt.plot(dims_array, times_bkw, 's-', label='BKW', color='crimson')

x_fit = np.linspace(start, end, 100)

plt.plot(x_fit, n4(x_fit, *params_pencil), color='royalblue', linestyle='--', label=r'Fit $\mathcal{O}(n^4)$')
plt.plot(x_fit, n7(x_fit, *params_bkw), color='crimson', linestyle='--', label=r'Fit $\mathcal{O}(n^7)$')

plt.yscale('log')
plt.xlabel("Dimension (n)")
plt.ylabel("Time (seconds, log-scale)")
plt.title("Runtime vs Dimension (log-scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
