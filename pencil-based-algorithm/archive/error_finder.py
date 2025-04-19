import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pencil import pencil_decompose, pencil_recompose

a, b = 0.6, 0.8  
n=8
c_vals = np.linspace(-10**n, 10**n, 50)  
d_vals = np.linspace(-10**n, 10**n, 50)

errors = np.zeros((len(d_vals), len(c_vals)))

for i, c in enumerate(c_vals):
    for j, d in enumerate(d_vals):
        f1 = [[a, b], [-b, a]]
        f2 = [[c, d], [-d, c]]
        tensor = np.stack([f1, f2], axis=2).astype(complex)
        re_tensor = pencil_recompose(pencil_decompose(tensor))
        err = np.linalg.norm(tensor - re_tensor)  
        errors[j, i] = err

delta_ac, delta_bd = np.meshgrid(c_vals - a, d_vals - b)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(delta_ac, delta_bd, errors, cmap='viridis', edgecolor='none')

ax.set_xlabel('c - a')
ax.set_ylabel('d - b')
ax.set_zlabel('Error')
ax.set_title('Pencil Decomposition Error Surface')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()
