
import numpy as np

# dit is for some foking reason tering instabiel maar dat was al te vermoeden
def diagonalize_basis(basis, tol=1e-10):
    results = []
    
    for X_i in basis:
        _,eigenvectors = np.linalg.eig(X_i)  
        eigenvectors=np.real_if_close(eigenvectors, tol=tol) 
        results.append(eigenvectors)
    
    return results