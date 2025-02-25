import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm

# Stap 1 van het algoritme
# We berekenen hier de afgeleide
"""
def find_derivative(tensor):
    dim = len(tensor) #cubic -> ok :)
    zeros = np.zeros(dim)
    [mode1,mode2,mode3] = [ttm(tensor,zeros,i).reshape(-1) for i in range(1,4)] # tenors already flattened
    return np.hstack((mode1, mode2, mode3))
"""
def unit_matrix(i, j, shape):
    M = np.zeros((shape,shape))
    M[i, j] = 1
    return M

def find_derivative(tensor):
    dim = tensor.shape[0] #cubic -> ok :)
    matrix=[]
    for k in range(1,4):
        for i in range(0,dim):
            for j in range(0,dim):
                u = unit_matrix(i,j,dim)
                matrix.append(np.vectorize(ttm(tensor,u,k)))
    
    return matrix.T

tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])

def print_frontal_slices(tensor):
    if len(tensor.shape) != 3:
        raise ValueError("Input must be a 3D tensor.")
    
    num_slices = tensor.shape[2]  

    for k in range(num_slices):
        print(f"Frontal Slice {k + 1}:\n", tensor[:, :, k], "\n")

tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])

print_frontal_slices(ttm(tensor,unit_matrix(0,0,2),1))