import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.optimize import curve_fit
from pencil import pencil_decompose, pencil_recompose

def random_orthogonal(n):
    H = np.random.randn(n, n)
    Q, _ = np.linalg.qr(H)
    return Q

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

start = 3
end = 50
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
