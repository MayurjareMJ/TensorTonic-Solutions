import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A without using np.transpose or .T
    """
    A = np.array(A)
    
    # Handle edge case: empty matrix
    if A.size == 0:
        return np.array(A)
    
    N, M = A.shape
    
    # Initialize result matrix
    result = np.empty((M, N), dtype=A.dtype)
    
    # Manual transpose
    for j in range(M):
        for i in range(N):
            result[j, i] = A[i, j]
    
    return result