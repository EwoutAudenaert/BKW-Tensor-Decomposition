import numpy as np
import matplotlib.pyplot as plt

def matrix_heatmap(matrix):
    """
    Displays a heatmap of a 2D matrix.

    Args:
        matrix (numpy.ndarray): 
            A 2D matrix to plot.
    """
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar() 
    plt.show()

def largest_modulus_coordinates_2d(matrix):
    """
    Finds, for each row, the column index of the largest modulus entry.

    Args:
        matrix (numpy.ndarray): 
            A 2D matrix.

    Returns:
        list of tuple: 
            List of (row, column) index pairs.
    """
    moduli = np.abs(matrix)
    coords = []
    for i in range(matrix.shape[0]):
        j = np.argmax(moduli[i, :])
        coords.append((i, int(j)))
    return coords

def largest_modulus_coordinates_3d(tensor):
    """
    Finds, for each frontal slice, the coordinates of the largest modulus entry.

    Args:
        tensor (numpy.ndarray): 
            A 3D tensor.

    Returns:
        list of tuple: 
            List of (i, j, k) index triples.
    """
    moduli = np.abs(tensor) # lineair --> O(n^3) I guess
    coords = []
    
    for i in range(tensor.shape[0]):
        j, k = np.unravel_index(np.argmax(moduli[i, :, :]), moduli[i, :, :].shape)
        coords.append((i, int(j), int(k)))  # Correct indexing
    return coords

def print_matrix(matrix):
    """
    Prints a 2D matrix in a formatted, readable text layout.

    Args:
        matrix (list or numpy.ndarray): 
            A 2D matrix to print.
    """
    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist() 
    cols = len(matrix[0])

    print(" " + "_" * (cols * 2+1 ))
    for row in matrix:
        print("|", " ".join(f"{elem}" for elem in row), "|")
    print(" " + "-" * (cols * 2 +1))

def array_depth(arr):
    """
    Computes the depth (level of nesting) of an array or list.

    Args:
        arr (list or numpy.ndarray): 
            Input array.

    Returns:
        int: 
            Depth of the array.
    """
    depth = 0
    while isinstance(arr, (list, np.ndarray)):
        depth += 1
        arr = arr[0] if len(arr) > 0 else None
    return depth

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

def ttm(X, m, mode):
    """
    Performs tensor-times-matrix multiplication along a specified mode.

    Args:
        X (numpy.ndarray): 
            A 3D tensor.
        m (numpy.ndarray): 
            A 2D matrix to multiply with.
        mode (int): 
            Mode along which to multiply (1, 2, or 3).

    Returns:
        numpy.ndarray: 
            Resulting 3D tensor after multiplication.
    """
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
    """
    Vectorizes a 3D tensor by stacking its frontal slices column-wise.

    Args:
        tensor (numpy.ndarray): 
            A 3D tensor.

    Returns:
        numpy.ndarray: 
            A 1D vector containing all tensor entries.
    """
    frontal_slices = [tensor[:, :, k].flatten(order='F') for k in range(tensor.shape[2])]
    return np.concatenate(frontal_slices)

def unvectorize_tensor(vector, shape):
    """
    Reconstructs a 3D tensor from its vectorized form.

    Args:
        vector (numpy.ndarray): 
            A 1D array representing the tensor.
        shape (tuple): 
            Target shape of the tensor (n, n, n).

    Returns:
        numpy.ndarray: 
            The reconstructed 3D tensor.
    """
    dim = shape[0]  
    tensor = np.zeros(shape)
    for k in range(shape[2]):
        tensor[:, :, k] = vector[k * dim**2 : (k + 1) * dim**2].reshape((dim, dim), order='F')
    return tensor

def print_latex_matrix(matrix,giveInt=False):
    """
    Prints a matrix in LaTeX bmatrix format.

    Args:
        matrix (numpy.ndarray): 
            Matrix to convert.
        giveInt (bool, optional): 
            Whether to convert entries to integers. Defaults to False.
    """
    if giveInt:
        matrix = matrix.astype(int)
    rows = [" & ".join(map(str, row)) for row in matrix]
    latex_matrix = "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"
    print(latex_matrix)

def print_frontal_slices(tensor):
    """
    Prints each frontal slice of a 3D tensor.

    Args:
        tensor (numpy.ndarray): 
            A 3D tensor.
    """
    if len(tensor.shape) != 3:
        raise ValueError("Input must be a 3D tensor.")
    
    num_slices = tensor.shape[2]  

    for k in range(num_slices):
        print(f"Frontal Slice {k + 1}:\n", tensor[:, :, k], "\n")

def helicoidal_tensor(n):
    """
    Generates a helicoidal pattern tensor of size (n, n, n).

    Args:
        n (int): 
            Dimension of the cubic tensor.

    Returns:
        numpy.ndarray: 
            A 3D tensor with ones along a helicoidal surface.
    """
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


def random_block_tensor(n=10, num_blocks=5, min_block=1, max_block=5):
    """
    Creates a 3D tensor with random cubic blocks filled with ones.

    Args:
        n (int, optional): 
            Dimension of the cubic tensor. Defaults to 10.
        num_blocks (int, optional): 
            Number of random blocks. Defaults to 5.
        min_block (int, optional): 
            Minimum block size. Defaults to 1.
        max_block (int, optional): 
            Maximum block size. Defaults to 5.

    Returns:
        numpy.ndarray: 
            A 3D tensor with random blocks.
    """
    tensor = np.zeros((n, n, n), dtype=float)
    max_block = min(max_block, n // 2)  # Make sure blocks are not bigger than n

    for _ in range(num_blocks):
        block_size = np.random.randint(min_block, max_block + 1)
        if n - block_size <= 0:
            continue
        x = np.random.randint(0, n - block_size)
        y = np.random.randint(0, n - block_size)
        z = np.random.randint(0, n - block_size)
        tensor[x:x+block_size, y:y+block_size, z:z+block_size] = 1
    return tensor



def round_matrix(m,digits=3):
    """
    Rounds the entries of a matrix to a specified number of digits.

    Args:
        m (list or numpy.ndarray): 
            Matrix to round.
        digits (int, optional): 
            Number of digits to round to. Defaults to 3.

    Returns:
        list: 
            Rounded matrix.
    """
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
    """
    Converts a matrix to a string formatted for Wolfram Language input.

    Args:
        m (list or numpy.ndarray): 
            Matrix to convert.
        digits (int, optional): 
            Number of digits to round. Defaults to 3.

    Returns:
        str: 
            Wolfram-compatible string.
    """
    return "{{" + "},{" .join(
        ",".join(str(cell) for cell in row)
        for row in round_matrix(m,digits)
    ) + "}}"

def plot_tensor(original_tensor,tensor,gridsize ,eps=0.1,surface=True,title1='Oorspronkelijke tensor',title2='Gereconstrueerde tensor'):
    """
    Plots the original and reconstructed tensors in 3D, showing points above a threshold.

    WARNING:
        - Only plots positive values! This is NOT a generic plot function! Read the code before you use it!
        - The helicoidal surface shown is a special function and may not behave as you expect.

    Args:
        original_tensor (numpy.ndarray): 
            The original tensor.
        tensor (numpy.ndarray): 
            The reconstructed tensor.
        gridsize (int): 
            Grid size used for surface plotting.
        eps (float, optional): 
            Threshold for displaying tensor entries. Defaults to 0.1.
        surface (bool, optional): 
            Whether to show the helicoidal surface. Defaults to True.
        title1 (str, optional): 
            Title for the first plot. Defaults to 'Oorspronkelijke tensor'.
        title2 (str, optional): 
            Title for the second plot. Defaults to 'Gereconstrueerde tensor'.
    """

    n=gridsize
    # parameters
    SHOW_SURFACE = surface
    ELEV = 30
    AZIM = 130
    grid_size = n

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(-2, 2, 100)
    U, V = np.meshgrid(u, v)
    # sinsoidal curve
    X = np.sin(U) * V
    Y = np.cos(U) * V
    Z = U

    # normalize coords
    X_discrete = ((X - X.min()) / (X.max() - X.min()) * (grid_size - 1)).astype(int)
    Y_discrete = ((Y - Y.min()) / (Y.max() - Y.min()) * (grid_size - 1)).astype(int)
    Z_discrete = ((Z - Z.min()) / (Z.max() - Z.min()) * (grid_size - 1)).astype(int)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(121, projection='3d')

    x_idx, y_idx, z_idx = np.where(abs(original_tensor) >= eps)
    ax.scatter(x_idx, y_idx, z_idx, color="red", s=10, alpha=0.8, label="Original Points")

    if SHOW_SURFACE:
        ax.plot_surface(X_discrete, Y_discrete, Z_discrete, color="blue", alpha=0.15, edgecolor='none')  

    ax.set_xlabel('X (Index)')
    ax.set_ylabel('Y (Index)')
    ax.set_zlabel('Z (Index)')
    ax.set_title(title1)
    ax.view_init(elev=ELEV, azim=AZIM)

    #reconstruction
    ax2 = fig.add_subplot(122, projection='3d')

    x_idx, y_idx, z_idx = np.where(abs(tensor) >= eps)  
    ax2.scatter(x_idx, y_idx, z_idx, color="green", s=10, alpha=0.8, label="Reconstructed Points")  

    if SHOW_SURFACE:
        ax2.plot_surface(X_discrete, Y_discrete, Z_discrete, color="blue", alpha=0.15, edgecolor='none') 

    ax2.set_xlabel('X (Index)')
    ax2.set_ylabel('Y (Index)')
    ax2.set_zlabel('Z (Index)')
    ax2.set_title(title2)
    ax2.view_init(elev=ELEV, azim=AZIM)
    plt.subplots_adjust(wspace=0.5) 

    plt.tight_layout()
    plt.show()


def plot_tensor_projection(tensor, title="Tensor Projection"):
    """
    Plots the 2D projection of a 3D tensor by summing along its third mode.

    Args:
        tensor (numpy.ndarray): 
            A 3D cubic tensor.
        title (str, optional): 
            Title of the plot. Defaults to "Tensor Projection".
    """
    if tensor.ndim != 3 or tensor.shape[0] != tensor.shape[1] or tensor.shape[1] != tensor.shape[2]:
        raise ValueError("Input tensor must be cubic (n x n x n)")

    tensor = np.abs(tensor)  
    matrix = np.sum(tensor, axis=2)

    plt.imshow(matrix, cmap='viridis', origin='lower', aspect='equal')
    plt.colorbar(label="Sum of the fiber")
    plt.title(title)
    plt.xlabel("Fiber index")
    plt.ylabel("Fiber index")
    plt.show()