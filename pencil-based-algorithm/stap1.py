import numpy as np

def find_mode_subspaces(tensor):
    dim = len(tensor) # fine since the tensor is cubic
    v1 = np.random.uniform(-1, 1, dim)
    v2 = np.random.uniform(-1, 1, dim)
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    pencil = np.tensordot(tensor,  matrix = np.column_stack((v1, v2)),axes=(0, 0))
    return pencil