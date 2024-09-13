import numpy as np

u = np.array([[ 3, -7], [-1,  4]])
v = np.array([[ 1/2,  -1/2], [-1/2, 1/2]])
w = np.array([[ -1/2, 1/2], [ -1/2, 1/2]])

# Scalar product
def scalar_product(img1, img2):
    return np.sum(img1 * img2)

# Norm
def norm(img1):
    return np.sqrt(np.sum(img1 ** 2))

# Part1
norm_u = norm(u)
norm_v = norm(v)
norm_w = norm(w)
u_dot_v = scalar_product(u, v)
u_dot_w = scalar_product(u, w)
v_dot_w = scalar_product(v, w)

print(f"||u|| = {norm_u}")
print(f"||v|| = {norm_v}")
print(f"||w|| = {norm_w}")
print(f"u · v = {u_dot_v}")
print(f"u · w = {u_dot_w}")
print(f"v · w = {v_dot_w}")

# Part2
# Since v and w are orthonormal, the coordinates can be defined below.
x1 = scalar_product(u, v)
x2 = scalar_product(u, w)

# Then the orthogonal projection is below.
f = (x1 * v) + (x2 * w)
print(f)
