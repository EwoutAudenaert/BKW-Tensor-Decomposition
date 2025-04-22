import numpy as np
import matplotlib.pyplot as plt

def matrix_heatmap(matrix):
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar() 
    plt.show()

def largest_modulus_coordinates_2d(matrix):
    moduli = np.abs(matrix)
    coords = []
    for i in range(matrix.shape[0]):
        j = np.argmax(moduli[i, :])
        coords.append((i, int(j)))
    return coords

def largest_modulus_coordinates_3d(tensor):
    moduli = np.abs(tensor) # lineair --> O(n^3) I guess
    coords = []
    
    for i in range(tensor.shape[0]):
        j, k = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append((i, int(j), int(k)))  # Correct indexing
    return coords

def print_matrix(matrix):
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist() 
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
small_tensor =  np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
deep_tensor = np.array([[
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],[
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]]
])

tensor1 =np.array([
    [[-4, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
tensor2 =np.array([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])


def plot_tensor():
    pass

def ttm(X, m, mode):
    match mode:
        case 1:
            X_mode = X.reshape(X.shape[0], -1) # time 1 ; space 1
            Y = m @ X_mode # time n^3 ; space n^2

            return Y.reshape(m.shape[0], X.shape[1], X.shape[2]) # time 1 ; space n^3

        case 2:
            X_mode = X.transpose(1, 0, 2).reshape(X.shape[1], -1)
            Y = m @ X_mode


            return Y.reshape(m.shape[0], X.shape[0], X.shape[2]).transpose(1, 0, 2)

        case 3:
            X_mode = X.transpose(2, 0, 1).reshape(X.shape[2], -1)
            Y = m @ X_mode

            return Y.reshape(m.shape[0], X.shape[0], X.shape[1]).transpose(1, 2, 0)
        

def vectorize_tensor(tensor):
    frontal_slices = [tensor[:, :, k].flatten(order='F') for k in range(tensor.shape[2])]
    return np.concatenate(frontal_slices)

def unvectorize_tensor(vector, shape):

    dim = shape[0]  
    tensor = np.zeros(shape)


    for k in range(shape[2]):
        tensor[:, :, k] = vector[k * dim**2 : (k + 1) * dim**2].reshape((dim, dim), order='F')
    
    return tensor
def print_latex_matrix(matrix,giveInt=False):
    if giveInt:
        matrix = matrix.astype(int)
    rows = [" & ".join(map(str, row)) for row in matrix]
    latex_matrix = "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"
    print(latex_matrix)

def print_frontal_slices(tensor):
    if len(tensor.shape) != 3:
        raise ValueError("Input must be a 3D tensor.")
    
    num_slices = tensor.shape[2]  

    for k in range(num_slices):
        print(f"Frontal Slice {k + 1}:\n", tensor[:, :, k], "\n")

ttm_test_t = np.array([[[1,2],[2,1]],[[3,0],[4,3]]])
ttm_test_m = np.array([[1,0],[0,0]])


def helicoidal_tensor(n):

    # surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-2, 2, 100)
    U, V = np.meshgrid(u, v)
    X = np.sin(U) * V
    Y = np.cos(U) * V
    Z = U

    # normalize coords
    X_discrete = ((X - X.min()) / (X.max() - X.min()) * (n - 1)).astype(int)
    Y_discrete = ((Y - Y.min()) / (Y.max() - Y.min()) * (n - 1)).astype(int)
    Z_discrete = ((Z - Z.min()) / (Z.max() - Z.min()) * (n - 1)).astype(int)

    tensor = np.zeros((n, n, n), dtype=float)

    # set points for surface
    for i in range(X_discrete.shape[0]):
        for j in range(X_discrete.shape[1]):
            x_idx, y_idx, z_idx = X_discrete[i, j], Y_discrete[i, j], Z_discrete[i, j]
            tensor[x_idx, y_idx, z_idx] = 1  
    return tensor

def round_matrix(m,digits=3):
    rounded = []
    for row in m:
        new_row = []
        for val in row:
            try:
                num = float(val)
                num = 0 if abs(num) < 1e-6 else round(num, digits)
                new_row.append(num)
            except:
                new_row.append(val)
        rounded.append(new_row)
    return rounded

def matrix_to_wolfram_string(m,digits=3):
    return "{{" + "},{" .join(
        ",".join(str(cell) for cell in row)
        for row in round_matrix(m,digits)
    ) + "}}"
