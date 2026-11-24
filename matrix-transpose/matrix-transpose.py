import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    m, n = len(A), len(A[0])
    B = [[0] * m for _ in range(n)]   # n×m
    for i in range(n):
        for j in range(m):
            B[i][j] = A[j][i]
    return np.array(B)