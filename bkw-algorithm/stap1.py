import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm

# Stap 1 van het algoritme
# We berekenen hier de afgeleide

def unit_matrix(i, j, shape):
    M = np.zeros((shape,shape))
    M[i, j] = 1
    return M

def print_frontal_slices(tensor):
    if len(tensor.shape) != 3:
        raise ValueError("Input must be a 3D tensor.")
    
    num_slices = tensor.shape[2]  

    for k in range(num_slices):
        print(f"Frontal Slice {k + 1}:\n", tensor[:, :, k], "\n")

def vectorize_tensor(tensor):
    frontal_slices = [tensor[:, :, k].flatten(order='F') for k in range(tensor.shape[0])]
    return np.concatenate(frontal_slices)

def find_derivative(tensor):
    dim = tensor.shape[0] #cubic -> ok :)
    matrix=[]
    for k in range(1,4):
        for i in range(0,dim):
            for j in range(0,dim):
                u = unit_matrix(i,j,dim)
                matrix.append(vectorize_tensor(ttm(tensor,u,k)).T)
    return matrix



