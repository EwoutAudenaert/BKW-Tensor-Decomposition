import numpy as np

def find_decomposition(results):
    decomposition = []  
    factors = []
    for diag_matrix, eigenvectors in results:
        factor = np.prod(np.diag(diag_matrix))  
        #in case of confusion, @ is mulitply
        A_i = eigenvectors @ diag_matrix @ np.linalg.inv(eigenvectors)
        factors.append(factor)
        decomposition.append(A_i)
    return decomposition,factor