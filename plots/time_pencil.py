import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.optimize import curve_fit
from exp_library import random_orthogonal
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pencil-based-algorithm')))
from pencil import pencil_decompose, pencil_recompose

# This script plots the runtime of the pencil-based decomposition algorithm as a function of tensor dimension.
# start: minimum dimension n of the cubic tensors (n x n x n) tested
# end: maximum dimension n of the cubic tensors (n x n x n) tested
# In this figure, start is set to 3 and end is set to 100.

start = 3
end = 100

def measure_time_pencil(n, repetitions=50):
    times = []
    for _ in range(repetitions):
        factor_matrices = [random_orthogonal(n) for _ in range(3)]
        tensor = pencil_recompose(factor_matrices)
        start = time.time()
        _ = pencil_decompose(tensor)
        end = time.time()
        times.append(end - start)
    return np.mean(times)


dims = range(start, end + 1)
dims_array = np.array(list(dims))

times_pencil = []

for n in dims:
    times_pencil.append(measure_time_pencil(n))

def n4(x, a):
    return a * x**4

params_pencil, _ = curve_fit(n4, dims_array, times_pencil)

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(12, 7))

plt.plot(dims_array, times_pencil, 'o-', label='Pencil-based', color='royalblue')

x_fit = np.linspace(start, end, 100)
plt.plot(x_fit, n4(x_fit, *params_pencil), color='royalblue', linestyle='--', label=r'Fit $\mathcal{O}(n^4)$')

plt.yscale('log')
plt.xlabel("Dimension (n)")
plt.ylabel("Time (seconds, log-scale)")
plt.title("Pencil-based Runtime vs Dimension (log-scale)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
