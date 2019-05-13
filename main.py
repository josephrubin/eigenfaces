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

    # Calculate the difference from the mean for each frame.
    mean = np.zeros(235 * 200).flatten()
    for img in images_vec:
        mean = np.add(mean, img)

    mean /= len(images_vec)
    #print(mean)

    difs = []
    for img in images_vec:
        i = img - mean
        difs.append(i)

    A = np.column_stack(difs)

    C = np.matmul(A.T, A)

    (values, vectors) = np.linalg.eig(C)

    vectors2 = []
    for v in vectors:
        vv = A.dot(v)
        vvv = vv / np.linalg.norm(vv)
        vectors2.append(vvv)

    print(vectors2)

    #print(values)

    #mean.resize(235, 200)
    #print(mean)
    #plt.matshow(mean)
    #plt.show()


main()