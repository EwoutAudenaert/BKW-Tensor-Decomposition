import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from library import ttm

# Stap 1 van het algoritme
# We berekenen hier de afgeleide
def find_derivative(tensor):
    dim = len(tensor) #cubic -> ok :)
    zeros = np.zeros(dim)
    [mode1,mode2,mode3] = [ttm(tensor,zeros,i).reshape(-1) for i in range(1,4)] # tenors already flattened
    return np.hstack((mode1, mode2, mode3))
