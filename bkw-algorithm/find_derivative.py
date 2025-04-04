import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm,unvectorize_tensor,vectorize_tensor

# Stap 1 van het algoritme
# We berekenen hier de afgeleide

def unit_matrix(i, j, shape):
    M = np.zeros((shape,shape)) # time n^2 ; space n^2
    M[i, j] = 1 # time 1 ; space 1
    return M # time 1 ; space 1

def find_derivative(tensor):
    dim = tensor.shape[0] # time 1 ; space 1
    matrix=[] # time 1 ; space 1
    for k in range(1,4): # time 1 ; space 1
        for i in range(0,dim): # time n ; space 1
            for j in range(0,dim): # time n ; space 1
                u = unit_matrix(i,j,dim) # time n^2 ; space n^2
                matrix.append(vectorize_tensor(ttm(tensor,u,k)).T) # time n^3 ; space n^3 --> because of ttm
    return matrix



tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
