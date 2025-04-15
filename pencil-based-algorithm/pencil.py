import numpy as np
from scipy.linalg import inv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import ttm

def pencil_decompose(tensor):
    #todo take random linear combination of slices
    n=len(tensor) # time O(1) ; space O(1)
    T1,T2 = tensor[:,:,0],tensor[:,:,1] # time O(n^2) ; space O(n^2)
    hiddenA = T1 @ inv(T2) # time O(n^3) ; space O(n^2)
    eigsA, A = np.linalg.eig(hiddenA)  # time O(n^3) ; space O(n^2)
    hiddenBt = inv(T1) @ T2 #B^-T diag(_) B^T with B a factor matrix ; time O(n^3) ; space O(n^2)
    eigsB, Bt = np.linalg.eig(hiddenBt)  # B^T ; time O(n^3) ; space O(n^2)
    Bt = Bt.conj() # time O(n^2) ; space O(1)
    B  = inv(Bt.T) # time: O(n^3) ; space O(n^2)
    Ctensor = ttm(ttm(tensor,inv(A),1),inv(B),2) # time O(n^4) ; space O(n^3)
    C = []
    for i in range(n): # time O(n^2) ; space O(n^2)
        C.append(Ctensor[i,i,:])
    #we need to transpose because np puts the row into the columns
    C = np.array(C).T.conj() # time O(n^2) ; space O(n^2)
    return [A,B,C]


def pencil_recompose(factor_matrices):
    n=len(factor_matrices[0])
    factor_tensor = np.zeros((n,n,n))
    for i in range(n):
        factor_tensor[i,i,i]=1
    [A,B,C] =factor_matrices
    return ttm(ttm(ttm(factor_tensor,A,1),B,2),C,3)

