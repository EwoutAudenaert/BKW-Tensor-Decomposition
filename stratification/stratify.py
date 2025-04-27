import numpy as np
import sys
import os
import tensorly as tl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,helicoidal_tensor,print_frontal_slices,print_matrix,matrix_heatmap,plot_tensor,ttmR,random_block_tensor
from find_derivative import find_derivative
from find_kernel import kernel
from scipy.linalg import inv,det
from tucker import tucker,mlp,tucker_recompose
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#tensor =  np.array([[[1,2,3],[2,1,-1],[0,7,-1]],[[3,0,0],[4,3,1],[9,0,5]],[[4,3,8],[9,0,5],[1,2,3]]])
#tensor = np.array([np.ones((3,3)),np.eye(3),-np.ones((3,3))])
#print_frontal_slices(tensor)
#core,factors = tucker(tensor)

n=10
tensor = helicoidal_tensor(n)
#core, factors = tucker(tensor) 
#print(core.shape)
core = tensor
def generate_invertible_matrices(n, count=3):
    matrices = []
    while len(matrices) < count:
        A = np.random.randn(n, n)
        if np.linalg.cond(A) < 1 / np.finfo(A.dtype).eps: 
            matrices.append(A)
    return matrices

[X,Y,Z] = generate_invertible_matrices(n,3)
scrambled_tensor = ttm(ttm(ttm(tensor,X,1),Y,2),Z,3) #todo add new ttms
#de_scrambled_tensor = ttm(ttm(ttm(scrambled_tensor,inv(X),1),inv(Y),2),inv(Z),3)
#plot_tensor(tensor,tensor,n)

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



def round_below_threshold_to_zero(tensor, threshold):
    tensor[tensor < threshold] = 0
    return tensor

from scipy.linalg import schur


result =[]
for m in basis:
    #eigenvalues instead of jordan normal form
    #vals,P = np.linalg.eig(m)
    #result.append(inv(P))
    T, P = schur(m, output='complex')  # T is upper triangular (almost Jordan form)
    result.append(inv(P).T)

[A,B,C]  = result


stratified_core = ttm(ttm(ttm(core,A,1),B,2),C,3)

#reconstructed_tensor  = tucker_recompose(stratified_core,factors)
reconstructed_tensor = stratified_core
#reconstructed_tensor= round_below_threshold_to_zero(reconstructed_tensor,1)
plot_tensor(tensor,reconstructed_tensor,n,False)
