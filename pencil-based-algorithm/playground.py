import numpy as np
from numpy.linalg import inv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import print_matrix,print_frontal_slices
from pencil import pencil_decompose,pencil_recompose

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bkw-algorithm')))
from bkw import bkw_recompose, bkw_decompose

def nearly_identical_orthogonal_tensor(n,eps=1e-6):
    Q1, _ = np.linalg.qr(np.random.randn(n, n))
    Q2, _ = np.linalg.qr(np.random.randn(n, n))    
    T1 = Q1 @ Q2.T
    # T2 should be almost T1, but we keep it ortho
    perturb = np.eye(n) + eps * np.random.randn(n, n)
    U, _ = np.linalg.qr(perturb)
    T2 = T1 @ U 
    tensor = np.zeros((n, n, n))
    tensor[:, :, 0] = T1
    tensor[:, :, 1] = T2
    for k in range(2, n):
        tensor[:, :, k] = np.eye(n)  
    return tensor

import numpy as np



#tensor = nearly_identical_orthogonal_tensor(3) #np.array([[[1,2],[2,1]],[[3,0],[4,3]]],dtype=complex)
#re_tensor = pencil_recompose(pencil_decompose(tensor))
#print_frontal_slices(tensor)
#print_frontal_slices(re_tensor)



"""
reps=100
errs = []
start =3
end =6
for n in range(start,end):
    acc_err = 0
    for rep in range(0,reps):
        #tensor = nearly_identical_orthogonal_tensor(n)
        #tensor  = np.random.rand(*(n,n,n))
        tensor = close_eigs_tensor(n,10**-12)
        re_tensor = pencil_recompose(pencil_decompose(tensor))   
        
        err  = np.sqrt(np.sum(np.abs(tensor - re_tensor)**2))#frobenius norm
        #err = np.max(np.abs(tensor - re_tensor))
        acc_err+=err
    errs.append(acc_err/reps)
"""

reps = 100
errs = []
c = 0.5  # fixed parameter
start =1
end =13
a_values = np.logspace(1, 12, num=12)  # decreasing a from 1e-1 to 1e-12
for a in a_values:
    T1 = np.array([[1, c], [-c, 1]])
    T2 = np.array([[1, c - a], [-(c - a), 1]])
    tensor = np.zeros((2, 2, 2))
    tensor[:, :, 0] = T1
    tensor[:, :, 1] = T2

    re_tensor = pencil_recompose(pencil_decompose(tensor))
    err = np.linalg.norm(tensor - re_tensor)  # Frobenius norm
    print("=============================================================")
    print_frontal_slices(tensor)
    print_frontal_slices(re_tensor)
    print_frontal_slices(tensor-re_tensor)

    errs.append(err )

print(errs)
exit()
#check error 1/sqrt(n)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

r = errs

# X-axis: dimensions from 2 to 7
dims = np.arange(start, end)

# Create plot
sns.set(style="whitegrid")
sns.lineplot(x=dims, y=r, marker="o")
plt.xlabel("Dimensions")
plt.ylabel("Error")
plt.title("Decomposition Error vs Dimensions")
plt.show()
