import numpy as np
from numpy.linalg import inv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from library import print_matrix,print_frontal_slices
from pencil import pencil_decompose,pencil_recompose

def nearly_identical_orthogonal_tensor(n):
    Q1, _ = np.linalg.qr(np.random.randn(n, n))
    Q2, _ = np.linalg.qr(np.random.randn(n, n))    
    T1 = Q1 @ Q2.T
    # T2 should be almost T1, but we keep it ortho
    eps = 1e-5
    perturb = np.eye(n) + eps * np.random.randn(n, n)
    U, _ = np.linalg.qr(perturb)
    T2 = T1 @ U 
    tensor = np.zeros((n, n, n))
    tensor[:, :, 0] = T1
    tensor[:, :, 1] = T2
    for k in range(2, n):
        tensor[:, :, k] = np.eye(n)  
    return tensor
tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]],dtype=complex)
re_tensor = pencil_recompose(pencil_decompose(tensor))
print_frontal_slices(tensor)
print_frontal_slices(re_tensor)

exit()
reps=10
errs = []
for n in range(2,7):
    acc_err = 0
    for rep in range(0,reps):
        tensor = nearly_identical_orthogonal_tensor(n)
        re_tensor = pencil_recompose(pencil_decompose(tensor))
        err = np.linalg.norm(tensor - re_tensor)  
        acc_err+=err
    errs.append(acc_err/reps)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example error array
r = np.random.rand(6)  # assuming dimensions 2 through 7 → 6 values

# X-axis: dimensions from 2 to 7
dims = np.arange(2, 8)

# Create plot
sns.set(style="whitegrid")
sns.lineplot(x=dims, y=r, marker="o")
plt.xlabel("Dimensions")
plt.ylabel("Error")
plt.title("Decomposition Error vs Dimensions")
plt.show()
