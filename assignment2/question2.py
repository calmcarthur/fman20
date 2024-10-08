import numpy as np
import matplotlib.pyplot as plt

def g(x):
    if abs(x) <= 1:
        return (6/7) * (1 - (abs(x)**2))  # example piece
    elif 1 < abs(x) <= 2:
        return (-6/7) * ((abs(x)**3) - (5 * (abs(x)**2)) + (8 * abs(x)) - 4)  # another piece
    else:
        return 0  # last piece

def F(x, f):
    total = 0
    for i in range(1, 9):  # summing from 1 to 9
        total += g(x - i) * f[i - 1]  # Adjusting index since f is 0-indexed
    return total

# Values for f and x range
f_values = [2, 1, 6, 3, 3, 4, 8, 11]
x_values = np.linspace(1, 11, 400)  # Generate 400 points from 0 to 20
F_values = [F(x, f_values) for x in x_values]  # Calculate F(x) for each x

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, F_values, label='F(x)', color='blue')
plt.title('Interpolation of F(x)')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid()
plt.legend()
plt.show()