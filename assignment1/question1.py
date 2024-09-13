import numpy as np

def row_calc(row):
    return (6 - row) / 6

def col_calc(col):
    return col / 6

def imageFunction(x, y):
    return x * np.sin(np.pi * y)

def makeCoordinateMatrices():

    col_matrix = np.fromfunction(lambda row,col: col_calc(col), (7, 7), dtype=float)
    row_matrix = np.fromfunction(lambda row,col: row_calc(row), (7, 7), dtype=float)

    # print(np.array2string(col_matrix, formatter={'float_kind':lambda x: f'{x:.2f}'}))
    # print(np.array2string(row_matrix, formatter={'float_kind':lambda x: f'{x:.2f}'}))
    
    return col_matrix, row_matrix

def sampleImage(matrix):

    sampled_matrix = imageFunction(matrix[0], matrix[1])

    # print(np.array2string(sampled_matrix, formatter={'float_kind':lambda x: f'{x:.2f}'}))

    return sampled_matrix

def quantizeImage(matrix, num_levels=32):
    
    scaled_image = matrix * (num_levels - 1)
    quantized_matrix = np.round(scaled_image).astype(int)
    
    return quantized_matrix

def main():
    final_matrix = sampleImage(makeCoordinateMatrices())
    quantized_matrix = quantizeImage(final_matrix)

    print(quantized_matrix)

if __name__ == "__main__":
    main()