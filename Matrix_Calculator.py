import numpy as np
from scipy.linalg import logm, eigvals
A_new = np.array([
    [0, 1, 0, 1, 1, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1],
    [0, 0, 0, 0, 1, 1.25, 1, 1.25, 1, 2, 1.25, 1],
    [0, 1, 0, 1.25, 1, 1, 0, 0, 2, 1.25, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1.25, 1, 0, 1],
    [0, 0, 0, 2, 0, 0, 0, 0, 1, 1.25, 0, 1],
    [1, 0, 1, 1.25, 1.25, 1.25, 0, 1, 0, 1, 1, 1.25],
    [0, 0, 0, 0, 1, 1, 0, 0, 1.25, 1.25, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1.25],
    [0, 0, 0, 1.25, 0, 0, 0, 0, 0, 0, 1.25, 0],
    [0, 0, 1, 1.25, 0, 1, 0, 2.5, 0, 1, 0, 0],
    [0, 0, 0, 1, 1.25, 0, 0, 0, 0, 0, 1, 0]
])
alpha = 0.5
I_new = np.eye(A_new.shape[0])
matrix_new = I_new - alpha * A_new
log_matrix_new = logm(matrix_new)
S_new = -log_matrix_new
eigenvalues = eigvals(alpha * A_new)
spectral_radius = np.max(np.abs(eigenvalues))
np.set_printoptions(suppress=True, precision=6)
print("Matrix S:")
print(S_new)
print("\nEigenvalues of A:")
print(eigenvalues)
print("\nSpectral radius of A:")
print(spectral_radius)
