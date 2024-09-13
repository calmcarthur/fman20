import numpy as np

one = np.array([[ 1, 0, -1],  [1, 0, -1], [0, 0, 0], [0, 0, 0]])
two = np.array([[ 1, 1, 1],  [1, 0, 1], [-1, -1, -1], [0, -1, 0]])
three = np.array([[ 0, 1, 0],  [1, 1, 1], [1, 0, 1], [1, 1, 1]])
four = np.array([[ 0, 0, 0],  [0, 0, 0], [1, 0, -1], [1, 0, -1]])
f = np.array([[ -1, 5, 2],  [12, 6, 6], [8, 2, 7], [-4, 3, 5]])

# Scalar product
def scalar_product(img1, img2):
    return np.sum(img1 * img2)

# Norm
def norm(img1):
    return np.sqrt(np.sum(img1 ** 2))

# Part1
s1 = scalar_product(one, two)
s2 = scalar_product(one, three)
s3 = scalar_product(one, four)
s4 = scalar_product(two, three)
s5 = scalar_product(two, four)
s6 = scalar_product(three, four)


print(f"1 · 2 = {s1}")
print(f"1 · 3 = {s2}")
print(f"1 · 4 = {s3}")
print(f"2 · 3 = {s4}")
print(f"2 · 4 = {s5}")
print(f"3 · 4 = {s6}")

# Part 2
x1 = round(scalar_product(f,(1/2 * one)),3)
x2 = round(scalar_product(f,(1/3 * two)),3)
x3 = round(scalar_product(f,(1/3 * three)),3)
x4 = round(scalar_product(f,(1/2 * four)),3)

print(x1, x2, x3, x4)



f = (x1 * (one*1/2)) + (x2 * (two*1/3)) + (x3 * (1/3*three)) + (x4 * (1/2*four))
print(f)