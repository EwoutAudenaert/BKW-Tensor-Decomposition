import numpy as np
import sys
import os
from sympy import Matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,print_frontal_slices,print_matrix
from find_derivative import find_derivative
from find_kernel import kernel
from scipy.linalg import inv
import tensorly as tl
from tensorly.decomposition import tucker

tensor =  np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
core, factors = tucker(tensor, rank=2) 

derivative = np.array(find_derivative(core)).T # Time: O(n^3)
basis = kernel(derivative) # Time: O(n^7)
def is_scaled_identity_matrix(A):
    if A.shape[0] != A.shape[1]:
        return False    
    diagonal_values = np.diag(A)
    if np.all(diagonal_values == diagonal_values[0]) and np.all(A == np.diag(diagonal_values)):
        return True
    return False

result =[]
for m in basis:
    P, J = Matrix(m).jordan_form()
    P = np.array(P.evalf(), dtype=np.float64)
    J = np.array(J.evalf(), dtype=np.float64)
    A_diag = np.diag(np.diag(J))
    if is_scaled_identity_matrix(A_diag):
        print("The tensor is already stratified.")
        exit()
    result.append(inv(P))
[A,B,C]  = result
stratified_core = ttm(ttm(ttm(tensor,A,1),B,2),C,3)
reconstructed_tensor = tl.tucker_to_tensor((stratified_core, factors))

print_frontal_slices(reconstructed_tensor)