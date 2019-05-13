import numpy as np
import collect
import compute
import vis
import matplotlib.pyplot as plt

def main():
    face_classes = []
    for i in range(1, 15 + 1):
        # Retrieve the images from the disk.
        images = collect.get_images(i)

        # Turn the images into vectors.
        faces = [img.flatten() for img in images]

        # Compute eigenfaces.
        mean, eigenfaces = compute.eigenfaces(faces)

        # Compute change-of-basis matrix.
        basis = np.stack(eigenfaces)

        # Compute face class.
        face_class = compute.face_class(mean, basis, faces)

        face_classes.append(face_class)

    # Retrieve the images from the disk.
    images = collect.get_images_all()
    print(len(images))

    # Turn the images into vectors.
    faces = [img.flatten() for img in images]

    # Compute eigenfaces.
    mean, eigenfaces = compute.eigenfaces(faces)

    # Compute change-of-basis matrix.
    basis = np.stack(eigenfaces)

    mface = faces[2]
    normal_mface = mface - mean

    coords = basis.dot(normal_mface)
    print(basis)
    print(normal_mface)
    print(coords)


main()