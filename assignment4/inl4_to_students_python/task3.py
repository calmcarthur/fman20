import numpy as np

# Fundamental matrix F
F = np.array([[2, 2, 4],
              [3, 3, 6],
              [-5, -10, -6]])

# Points in image 1
a1 = np.array([2, 8, 1])
a2 = np.array([-3, 0, 1])
a3 = np.array([1, 16, 1])

# Points in image 2
b1 = np.array([1, 2, 1])
b2 = np.array([2, 3, 1])
b3 = np.array([3, 1, 1])

# Function to compute b^T F a
def epipolar_constraint(b, F, a):
    return np.dot(b.T, np.dot(F, a))

# Check all combinations
results = {
    "b1_a1": epipolar_constraint(b1, F, a1),
    "b1_a2": epipolar_constraint(b1, F, a2),
    "b1_a3": epipolar_constraint(b1, F, a3),
    "b2_a1": epipolar_constraint(b2, F, a1),
    "b2_a2": epipolar_constraint(b2, F, a2),
    "b2_a3": epipolar_constraint(b2, F, a3),
    "b3_a1": epipolar_constraint(b3, F, a1),
    "b3_a2": epipolar_constraint(b3, F, a2),
    "b3_a3": epipolar_constraint(b3, F, a3)
}

for key, value in results.items():
    print(f"{key}: {value}")