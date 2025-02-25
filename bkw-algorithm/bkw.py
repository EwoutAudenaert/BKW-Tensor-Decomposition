from stap1 import find_derivative
from stap2 import nulruimte
import numpy as np
from library import print_tensor
"""
tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
"""


tensor = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
vecs =find_derivative(tensor)
matrix = np.column_stack(vecs)
basis = nulruimte(matrix)

print_tensor(matrix)