import numpy as np

def find_decomposition(results):
    decomposition = []  
    
    for diag_matrix, eigenvectors in results:
        factor = np.prod(np.diag(diag_matrix))  
        #in case of confusion, @ is mulitply
        A_i = eigenvectors @ diag_matrix @ np.linalg.inv(eigenvectors)
        decomposition.append(factor,A_i)
    return decomposition
