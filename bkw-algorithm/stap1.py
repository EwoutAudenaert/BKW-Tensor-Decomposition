import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,unvectorize_tensor,vectorize_tensor

# Stap 1 van het algoritme
# We berekenen hier de afgeleide

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
                matrix.append(vectorize_tensor(ttm(tensor,u,k)).T)
    return matrix



tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
