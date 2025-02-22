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
def find_derivative(tensor):
    dim = tensor.shape[0] #cubic -> ok :)
    zeros = np.zeros((dim, dim))  
    
    mode1 = ttm(tensor, zeros, 1).reshape(-1)
    mode2 = ttm(tensor, zeros, 2).reshape(-1)
    mode3 = ttm(tensor, zeros, 3).reshape(-1)
    
    return np.hstack((mode1, mode2, mode3))


tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])

print(find_derivative(tensor))