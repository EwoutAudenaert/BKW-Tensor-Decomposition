
import numpy as np
import scipy.linalg

def diagonalize_basis(basis):
    results = []
    
    for X_i in basis:
        eigenvalues, eigenvectors = np.linalg.eig(X_i)        
        diag_matrix = np.diag(eigenvalues)
        results.append((diag_matrix, eigenvectors))
    
    return results



basis = [
    np.array([[2, 1], [1, 2]]),  
    np.array([[3, 2], [2, 3]])  ]


diagonalized = diagonalize_basis(basis)
for i, (Lambda, V) in enumerate(diagonalized):
    print(f"Matrix {i+1}:")
    print("Eigenvalues (Diagonal Matrix):\n", Lambda)
    print("Eigenvectors:\n", V)
    print()
