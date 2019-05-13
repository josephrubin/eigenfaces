import numpy as np
import collect
import matplotlib.pyplot as plt

def main():
    # Retrieve the images from the disk.
    images = collect.get_images(1)

    # Crop our images to center the face.
    images_cropped = [img[5:240, 90:290] for img in images]

    # Turn the images into vectors.
    images_vec = [img.flatten() for img in images_cropped]

    # Compute the mean.
    mean = np.zeros(235 * 200).flatten()
    for img in images_vec:
        mean = np.add(mean, img)

    mean /= len(images_vec)

    # Calculate the difference from the mean for each frame.
    difs = []
    for img in images_vec:
        i = img - mean
        difs.append(i)

    # Compute A matrix.
    A = np.column_stack(difs)

    # Compute (A transpose)(A)
    C = np.matmul(A.T, A)

    # Compute eigenvalues of (A transpose)(A).
    (values, vectors) = np.linalg.eig(C)

    # Compute best eigenvectors of (A)(A transpose).
    vectors2 = []
    for v in vectors:
        vv = A.dot(v)
        vvv = vv / np.linalg.norm(vv)
        vectors2.append(vvv)

    #print(vectors2)

    """
    vectors3 = [((v * 100) + 200) for v in vectors2]
    for v in vectors3:
        v.resize(235, 200)
        print(v.shape)
        plt.matshow(v)
        plt.show()
    """

    #print(values)

    #mean.resize(235, 200)
    #print(mean)
    #plt.matshow(mean)
    #plt.show()


main()