import numpy as np
import scipy.linalg as la

def qz_joint_eigenvalue_decomposition(pencil):
    A, B = pencil[..., 0], pencil[..., 1]
    _, Z, _, _ = la.qz(A, B)
    return Z  # Z = eigenvectors