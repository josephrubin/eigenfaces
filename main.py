import numpy as np
from scipy.spatial import distance
from scipy import ndimage
import collect
import compute
import vis
import random

def main():
    # First, compute the face classes for each person.
    # We'll leave out the first image for each person,
    # since this step represents the "training", and
    # we would like to use the first image as the
    # "test" for our system at the end.
    face_classes = []
    faces_individual = []
    faces_all = []
    test_faces = []
    for i in range(1, 15 + 1):
        # Retrieve the cropped images from the disk.
        images_ = collect.get_images(i)
        images=[]
        for i in images_:
            images.append(modify(i))

        #images = images[0:3] + images[4:6] + images[6:]

        # Turn the images into vectors.
        faces = [img.flatten() for img in images]

        # Save the first face from each set for the test.
        test_faces.append(faces[0])

        # The rest of the faces will be our training data.
        training_faces = faces[1:]
        faces_individual.append(training_faces)
        faces_all.extend(training_faces)

    print("Images loaded.")

    # Now, we want to compute the change-of-basis matrix
    # over the entire data set (again, leaving off the first
    # image from each set). This will represent our subspace
    # which is necessary for classifying new images.

    # Compute eigenfaces.
    mean, eigenfaces = compute.eigenfaces(faces_all)
    print("Eigenfaces computed.")

    # Compute change-of-basis matrix.
    basis = np.stack(eigenfaces)

    # Compute the face classes.
    for individual in faces_individual:
        face_class = compute.face_class(mean, basis, individual)
        face_classes.append(face_class)

        print("Face class {} computed.")

    num_right = 0
    for j, test_face in enumerate(test_faces):
        print("_________________________")
        print("test: " + str(j + 1))
        normal_face = test_face - mean
        coords = basis.dot(normal_face)

        min_class = None
        min_mag = None
        for i, face_class in enumerate(face_classes):
            normm = c(face_class, coords)
            diff = face_class - coords
            mag = normm#np.linalg.norm(diff)
            #print(mag)
            if (min_mag is None or mag < min_mag):
                min_mag = mag
                min_class = i
        print("Best class match: " + (str(min_class + 1)))
        if min_class == j:
            num_right += 1
    print("Correct: " + str(num_right))


def c(v1, v2):
    return distance.sqeuclidean(v1, v2)


def modify(face):
    n = (np.roll(face, 3) + np.roll(face, -3)) / 2
    if random.randint(0, 100) == 5:
        vis.display(n)
    return n


main()