import numpy as np
from scipy.spatial import distance

import collect
import compute
from const import *
import vis


__author__ = "Joseph, Fernando"


def main():
    # Check const.py to toggle VERBOSE_MODE.
    run_trials(TEST_FACE_INDEX, NUM_EIGENFACES)


def run_trials(test_face_index, num_eigenfaces):
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
        images = collect.get_images(i)

        # Turn the images into vectors.
        faces = [img.flatten() for img in images]

        # Save one face from each set for the test.
        test_faces.append(faces[test_face_index])

        # The rest of the faces will be our training data.
        training_faces = faces[0:test_face_index]\
            + faces[test_face_index + 1:]
        faces_individual.append(training_faces)
        faces_all.extend(training_faces)

    if VERBOSE_MODE:
        print("Images loaded.")

    # Compute eigenfaces.
    mean, eigenfaces = compute.eigenfaces(faces_all, num_eigenfaces)
    if VERBOSE_MODE:
        print("Eigenfaces computed.")

    # Compute change-of-basis (subspace projection) matrix.
    basis = np.stack(eigenfaces)

    # Compute the face classes.
    for individual in faces_individual:
        face_class = compute.face_class(mean, basis, individual)
        face_classes.append(face_class)
    if VERBOSE_MODE:
        print("Face classes computed.")

    num_right = 0
    for j, test_face in enumerate(test_faces):
        if VERBOSE_MODE:
            print("Trial: " + str(j + 1), end='')
        normal_face = test_face - mean
        coords = basis.dot(normal_face)

        min_class = None
        min_mag = None
        for i, face_class in enumerate(face_classes):
            mag = distance.sqeuclidean(face_class, coords)

            if min_mag is None or mag < min_mag:
                min_mag = mag
                min_class = i

        if VERBOSE_MODE:
            print(", best class match: " + (str(min_class + 1)))
        if min_class == j:
            num_right += 1

    if VERBOSE_MODE:
        print("\nNumber correct: " + str(num_right) + " / " + str(len(test_faces)))
        print("Percentage: " + str(100 * num_right / len(test_faces)) + "%")
        print("Eigenfaces used: " + str(num_eigenfaces))

    return num_right, num_right / len(test_faces)


if __name__ == "__main__":
    main()
