import numpy as np
import math

def imageFunction(x, y):
    return x * np.sin(np.pi * y)


print(imageFunction(1/6,1))