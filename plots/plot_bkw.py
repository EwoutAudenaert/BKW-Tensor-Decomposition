import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorly as tl
from tensorly.decomposition import tucker

# parameters
SHOW_SURFACE = True
ELEV = 30
AZIM = 130
grid_size = 50

# surface
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(-2, 2, 100)
U, V = np.meshgrid(u, v)
X = np.sin(U) * V
Y = np.cos(U) * V
Z = U




# normalize coords
X_discrete = ((X - X.min()) / (X.max() - X.min()) * (grid_size - 1)).astype(int)
Y_discrete = ((Y - Y.min()) / (Y.max() - Y.min()) * (grid_size - 1)).astype(int)
Z_discrete = ((Z - Z.min()) / (Z.max() - Z.min()) * (grid_size - 1)).astype(int)

tensor = np.zeros((grid_size, grid_size, grid_size), dtype=float)

# set points for surface
for i in range(X_discrete.shape[0]):
    for j in range(X_discrete.shape[1]):
        x_idx, y_idx, z_idx = X_discrete[i, j], Y_discrete[i, j], Z_discrete[i, j]
        tensor[x_idx, y_idx, z_idx] = 1  

# tucker
rank = [10, 10, 10] 
core, factors = tucker(tl.tensor(tensor), rank=rank)
reconstructed_tensor = tl.tucker_to_tensor((core, factors))


fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(121, projection='3d')

x_idx, y_idx, z_idx = np.where(tensor == 1)
ax.scatter(x_idx, y_idx, z_idx, color="red", s=10, alpha=0.8, label="Original Points")

if SHOW_SURFACE:
    ax.plot_surface(X_discrete, Y_discrete, Z_discrete, color="blue", alpha=0.15, edgecolor='none')  

ax.set_xlabel('X (Index)')
ax.set_ylabel('Y (Index)')
ax.set_zlabel('Z (Index)')
ax.set_title('Oorspronkelijke tensor')
ax.view_init(elev=ELEV, azim=AZIM)
 
#reconstruction
ax2 = fig.add_subplot(122, projection='3d')

x_idx, y_idx, z_idx = np.where(reconstructed_tensor >= 0.5)  
ax2.scatter(x_idx, y_idx, z_idx, color="green", s=10, alpha=0.8, label="Reconstructed Points")  

if SHOW_SURFACE:
    ax2.plot_surface(X_discrete, Y_discrete, Z_discrete, color="blue", alpha=0.15, edgecolor='none') 

ax2.set_xlabel('X (Index)')
ax2.set_ylabel('Y (Index)')
ax2.set_zlabel('Z (Index)')
ax2.set_title('Gereconstrueerde tensor (Tucker Decomposition)')
ax2.view_init(elev=ELEV, azim=AZIM)
plt.subplots_adjust(wspace=0.5) 

plt.tight_layout()
plt.show()