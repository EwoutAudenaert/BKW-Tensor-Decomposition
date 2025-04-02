def flatten_and_solve(tensor, Z):
    Z_inv = np.linalg.pinv(Z)  # pseudo-inverse
    tensor_flat = tensor.reshape(Z.shape[0], -1)  # flattening 
    result = Z_inv @ tensor_flat  # @ is weer vermenigvuldigen eigenlijk solven we hier
    return result.reshape(tensor.shape)  # moet nog gereshped worden