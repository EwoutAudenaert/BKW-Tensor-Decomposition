
import numpy as np

# dit is for some foking reason tering instabiel maar dat was al te vermoeden
def diagonalize_basis(basis):
    results = []
    
    for X_i in basis:
        _,eigenvectors = np.linalg.eig(X_i)        
        results.append(eigenvectors)
    
    return results



basis = [
    np.array([[2, 1], [1, 2]]),  
    np.array([[3, 2], [2, 3]]),
    np.array([[4, 1], [1, 3]])  ]


diagonalized = diagonalize_basis(basis)
for i, (Lambda, V) in enumerate(diagonalized):
    print(f"Matrix {i+1}:")
    print("Eigenvalues (Diagonal Matrix):\n", Lambda)
    print("Eigenvectors:\n", V)
    print()
