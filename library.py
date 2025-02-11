import numpy as np
# numpy matrices enkel
def print_matrix(matrix):
    matrix.flatten()
    rows = len(matrix)
    cols = len(matrix[0])

    print(" " + "_" * (cols * 2+1 ))
    for row in matrix:
        print("|", " ".join(f"{elem}" for elem in row), "|")
    print(" " + "-" * (cols * 2 +1))

