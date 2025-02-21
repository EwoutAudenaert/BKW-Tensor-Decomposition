import numpy as np
def print_matrix(matrix):
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist() 
    rows = len(matrix)
    cols = len(matrix[0])

    print(" " + "_" * (cols * 2+1 ))
    for row in matrix:
        print("|", " ".join(f"{elem}" for elem in row), "|")
    print(" " + "-" * (cols * 2 +1))

def array_depth(arr):
    depth = 0
    while isinstance(arr, (list, np.ndarray)):
        depth += 1
        arr = arr[0] if len(arr) > 0 else None
    return depth

assert array_depth([]) == 1
assert array_depth([[1,2,3],[5,6,7]]) == 2
assert array_depth([[[]]])==3

def print_tensor(A,m_depth=None):
    if m_depth is None:
        m_depth = array_depth(A)
    depth = array_depth(A)
    if depth == 2:
        return print_matrix(A)
    for i in range(0,len(A)):
        print("slice : " + "_ "*(depth-1) + str(i+1) + " _"*(m_depth -(depth)) )
        print_tensor(A[i],m_depth)
        print("\n")

deep_tensor = np.array([[
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],[
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
])

tensor =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])


#print_tensor(deep_tensor)

def plot_tensor():
    pass

def ttm(X, Zeros, mode):
    match mode:
        case 1:
            X_mode = X.reshape(X.shape[0], -1)
            Y = Zeros @ X_mode
            return Y.reshape(Zeros.shape[0], X.shape[1], X.shape[2])

        case 2:
            X_mode = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
            Y = Zeros @ X_mode
            return Y.reshape(Zeros.shape[0], X.shape[0], X.shape[2]).transpose(1, 0, 2)

        case 3:
            X_mode = X.transpose(2, 0, 1).reshape(X.shape[2], -1)
            Y = Zeros @ X_mode
            return Y.reshape(Zeros.shape[0], X.shape[0], X.shape[1]).transpose(1, 2, 0)

        case _:
            raise ValueError("Mode must be 1, 2, or 3.")