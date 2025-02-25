
import numpy as np

# dit is for some foking reason tering instabiel maar dat was al te vermoeden
def diagonalize_basis(basis):
    results = []
    
    for X_i in basis:
        _,eigenvectors = np.linalg.eig(X_i)        
        results.append(eigenvectors)
    
    return results