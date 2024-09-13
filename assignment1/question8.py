import scipy
import numpy as np
import os
import matplotlib.pyplot as plt

# Scalar product
def scalar_product(img1, img2):
    return np.sum(img1 * img2)

def projection(img, basis):
    # Initialize an empty result matrix and error norm
    img2 = np.zeros(img.shape)
    r = 0

    for x in range(basis.shape[2]):
        e_value = basis[:,:,x]
        coefficient = scalar_product(img, e_value)
        img2 += coefficient * e_value

    r = np.linalg.norm(np.abs(img - img2))

    return img2 ,r

def displayBases(bases1, bases2, bases3):
    _,axs = plt.subplots(4,3)
    axs[0][0].imshow(bases1[:,:,0])
    axs[1][0].imshow(bases1[:,:,1])
    axs[2][0].imshow(bases1[:,:,2])
    axs[3][0].imshow(bases1[:,:,3])
    axs[0][1].imshow(bases2[:,:,0])
    axs[1][1].imshow(bases2[:,:,1])
    axs[2][1].imshow(bases2[:,:,2])
    axs[3][1].imshow(bases2[:,:,3])
    axs[0][2].imshow(bases3[:,:,0])
    axs[1][2].imshow(bases3[:,:,1])
    axs[2][2].imshow(bases3[:,:,2])
    axs[3][2].imshow(bases3[:,:,3])
    axs[0, 0].set_title("Basis 1")
    axs[0, 1].set_title("Basis 2")
    axs[0, 2].set_title("Basis 3")
    plt.show()
    return

def displayStacks(general, faces):
    _,axs = plt.subplots(2,2)
    axs[0, 0].imshow(general[:, :, 0])
    axs[0, 0].set_title("General Image 1")
    axs[1, 0].imshow(general[:, :, 1])
    axs[1, 0].set_title("General Image 2")
    axs[0, 1].imshow(faces[:, :, 2])
    axs[0, 1].set_title("Face Image 1")
    axs[1, 1].imshow(faces[:, :, 3])
    axs[1, 1].set_title("Face Image 2")
    plt.tight_layout()
    plt.show()
    return

def main():

    data = scipy.io.loadmat('../inl1_to_students_python/assignment1bases.mat')
    bases = data['bases']
    stacks = data['stacks']

    general_stack = stacks[0,0] # 19 x 19 x 400
    faces_stack = stacks[0,1] # 19 x 19 x 400

    bases1 = bases[0,0] # 19 x 19 x 4
    bases2 = bases[0,1] # 19 x 19 x 4
    bases3 = bases[0,2] # 19 x 19 x 4

    # Display Bases
    # displayBases(bases1,bases2,bases3)

    # Display Stacks
    # displayStacks(general_stack, faces_stack)

 
    # Initialize lists to store error norms
    general_error_norms_b1 = []
    general_error_norms_b2 = []
    general_error_norms_b3 = []

    face_error_norms_b1 = []
    face_error_norms_b2 = []
    face_error_norms_b3 = []

    # General Stack: Project each general image onto all three bases
    for i in range(general_stack.shape[2]):
        img = general_stack[:, :, i]
        
        _, err1 = projection(img, bases1)
        _, err2 = projection(img, bases2)
        _, err3 = projection(img, bases3)
        
        general_error_norms_b1.append(err1)
        general_error_norms_b2.append(err2)
        general_error_norms_b3.append(err3)

    # Face Stack: Project each face image onto all three bases
    for i in range(faces_stack.shape[2]):
        img = faces_stack[:, :, i]
        
        _, err1 = projection(img, bases1)
        _, err2 = projection(img, bases2)
        _, err3 = projection(img, bases3)
        
        face_error_norms_b1.append(err1)
        face_error_norms_b2.append(err2)
        face_error_norms_b3.append(err3)

    # Compute the mean error norms
    mean_general_error_b1 = np.mean(general_error_norms_b1)
    mean_general_error_b2 = np.mean(general_error_norms_b2)
    mean_general_error_b3 = np.mean(general_error_norms_b3)

    mean_face_error_b1 = np.mean(face_error_norms_b1)
    mean_face_error_b2 = np.mean(face_error_norms_b2)
    mean_face_error_b3 = np.mean(face_error_norms_b3)

    # Print the mean error norms
    print("Mean Error Norms for General Images:")
    print(f"Basis 1: {mean_general_error_b1}")
    print(f"Basis 2: {mean_general_error_b2}")
    print(f"Basis 3: {mean_general_error_b3}")

    print("\nMean Error Norms for Face Images:")
    print(f"Basis 1: {mean_face_error_b1}")
    print(f"Basis 2: {mean_face_error_b2}")
    print(f"Basis 3: {mean_face_error_b3}")

    return

if __name__ == "__main__":
    main()