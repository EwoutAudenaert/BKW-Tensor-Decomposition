from find_eigenvalues import diagonalize_basis
import numpy as np
from scipy.linalg import inv
import numpy as np
import sys
import os
from bkw import bkw_decompose,bkw_recompose


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from find_derivative import find_derivative
from find_kernel import kernel
from library import ttm,print_frontal_slices,largest_modulus_coordinates_3d,print_matrix

def random_orthogonal(n):
    H = np.random.randn(n, n)
    Q, _ = np.linalg.qr(H)
    return Q
n=10
factor_matrices = [random_orthogonal(n) for _ in range(0,3)]
tensor = bkw_recompose([i for i in range(1,n+1)],factor_matrices)
factors, re_factor_matrices = bkw_decompose(tensor)