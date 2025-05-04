import numpy as np
import matplotlib.pyplot as plt


def sample_from_unit_circle():
    """
    Sample a complex number from the unit circle.
    """
    angle = np.random.uniform(0, 2*np.pi)
    return np.exp(1j * angle)

def interpolate_matrices(matrix_0, matrix_1, alpha):
    t1 = sample_from_unit_circle()
    t2 = sample_from_unit_circle()
    interpolated_matrix = (1 - alpha) * matrix_0 + alpha * matrix_1
    interpolated_matrix[0, 0] = t1
    interpolated_matrix[3, 3] = t2
    return interpolated_matrix



def plot_eigenvalues(eigenvalues_list, title):
    for eigenvalues in eigenvalues_list:
        plt.scatter(eigenvalues.real, eigenvalues.imag, marker='.', s=1, color='black')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(title)
    plt.grid(True)
    plt.show()

matrix_0 = np.array([
    [1, -1, -1j, -1j, -1j],
    [-1, 1, 1j, -1, 1j],
    [1, 1j, 0, 1, 0],
    [-1, -1j, -1, 2, -1],
    [0, 0, 1, 0, 1]
])

matrix_1 = np.array([
    [-1, 1, 1, 1j, -1],
    [1, 1, 0, -1, -1j],
    [-1, 1j, 3, -1, 1j],
    [1j, -1, -1, 0, -1],
    [1, 1, 4, 1, -1]
])

# Parameters
num_interpolations = 2
num_samples_per_interpolation = 2000
eigenvalues_list = []

for i in range(num_interpolations):
    alpha = i / num_interpolations
    for _ in range(num_samples_per_interpolation):
        interpolated_matrix = interpolate_matrices(matrix_0, matrix_1, alpha)
        eigenvalues = np.linalg.eigvals(interpolated_matrix)
        eigenvalues_list.append(eigenvalues)

plot_eigenvalues(eigenvalues_list, "Interpolated Eigenvalues with Sampling")
