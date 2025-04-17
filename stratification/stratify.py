import numpy as np
import sys
import os
import tensorly as tl

from sympy import Matrix, nsimplify
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,print_frontal_slices,print_matrix,helicoidal_tensor,matrix_heatmap
from find_derivative import find_derivative
from find_kernel import kernel
from scipy.linalg import inv,det
#from tucker import tucker
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#tensor =  np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
zn=10
tensor = helicoidal_tensor(n)

core,factors = tucker(tensor)
#print([f.shape for f in factors])
factors = [f.T  for f in factors]
reconstructed_tensor = tl.tucker_tensor.tucker_to_tensor((core, factors[::-1]))

matrix_heatmap(reconstructed_tensor[:,:,5])
exit()
def generate_invertible_matrices(n, count=3):
    matrices = []
    while len(matrices) < count:
        A = np.random.randn(n, n)
        if np.linalg.cond(A) < 1 / np.finfo(A.dtype).eps: 
            matrices.append(A)
    return matrices

[X,Y,Z] = generate_invertible_matrices(n,3)
scrambled_tensor = ttm(ttm(ttm(tensor,X,1),Y,2),Z,3) #todo add new ttms

#some tucker ask rank to get a threshold for the singular value threshold
# and that threshold is related in the rank, the multilineair rank is related to the number of singular value of the tensor
#same as singular values in matrix rank is number of positive singular values
# change take the first 20 just replace by below this threshold
#make own tucker. not dleto!!!!!
#this does not work because it will always keep the whole kernel
core, factors = tucker(scrambled_tensor) 

#have a look at of BKW heuristic version instead
# trivial solution dimension is 2 => hardcode step 1. If you have the derivative map and 
# X, Y,Z are all scalars then  x+y+z = 0 => 2 degrees of freedom

derivative = np.array(find_derivative(core)).T # Time: O(n^3)
basis = kernel(derivative) # Time: O(n^7)

def is_scaled_identity_matrix(A):
    if A.shape[0] != A.shape[1]:
        return False    
    diagonal_values = np.diag(A)
    if np.all(diagonal_values == diagonal_values[0]):
        return True
    return False

result =[]
for m in basis:
    m_sym = Matrix(m.round(decimals=1)).applyfunc(lambda x: nsimplify(x, rational=True))

    P, J = Matrix(m_sym).jordan_form()
    P = np.array(P.evalf(), dtype=np.complex128)
    J = np.array(J.evalf(), dtype=np.complex128)

    A_diag = np.diag(np.diag(J))
    
    result.append(inv(P))

[A,B,C]  = result
stratified_core = ttm(ttm(ttm(tensor,A,1),B,2),C,3)
# factor matrices ttm1 ttm2 ttm3 with the transposed matrices
#reconstructed_tensor = tucker_recompose(stratified_core, factors)
reconstructed_tensor = tl.tucker_tensor.tucker_to_tensor((core, factors[::-1]))

def round_below_threshold_to_zero(tensor, threshold):
    tensor[tensor < threshold] = 0
    return tensor
print_matrix(tensor[:,:,5])
reconstructed_tensor= round_below_threshold_to_zero(reconstructed_tensor,1)



# parameters
SHOW_SURFACE = True
ELEV = 30
AZIM = 130
grid_size = n

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