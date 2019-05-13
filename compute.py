import numpy as np

from const import *


__author__ = "Joseph, Fernando"


def eigenfaces(faces, num_eigenfaces):
    """ Given a set of face vectors (faces), compute the eigenfaces -

    a set of the most important (associated with the highest eigenvalues)
    eigenvectors of the covariance matrix (AA^T). Also return the mean face.
    """

    # Compute the mean.
    mean = face_mean(faces)

    # Calculate the difference from the mean for each face.
    diffs = []
    for face in faces:
        diff = face - mean
        diffs.append(diff)

    # Compute the normalized faces matrix.
    A = np.column_stack(diffs)

    # Compute (A transpose)(A), a matrix which will allow
    # us to find the eigenvectors of the covariance matrix.
    C = np.matmul(A.T, A)

    # Compute eigenvalues of (A transpose)(A).
    # Using eigh rather than eig is efficient when working with
    # a symmetric matrix.
    (eigen_values, eigen_vectors) = np.linalg.eigh(C)

    # eigen_vectors (matrix) contains one in each column, but iteration
    # happens, by default, through the rows. So we should transpose
    # the matrix before continuing.
    combined = list(zip(eigen_vectors.T, eigen_values))
    combined.sort(key=lambda t: t[1], reverse=True)
    combined_best = combined[0:num_eigenfaces]
    eigen_vectors_best = [t[0] for t in combined_best]

    # Compute best eigenvectors of (A)(A transpose).
    # These will be our eigenfaces.
    eigen_faces = []
    for eigen_vector in eigen_vectors_best:
        eigen_face_scaled = A.dot(eigen_vector)
        eigen_face = eigen_face_scaled / np.linalg.norm(eigen_face_scaled)
        eigen_faces.append(eigen_face)

    return mean, eigen_faces


def face_mean(faces):
    mean = np.zeros(IMG_SIZE).flatten()
    for face in faces:
        mean = np.add(mean, face)
    mean /= len(faces)

    return mean


def face_class(mean, basis, faces):
    # Find coords of each normalize face w/r/t
    # the eigenfaces.
    fclass = np.zeros(basis.shape[0]).flatten()
    for face in faces:
        diff = face - mean
        coords = basis.dot(diff)

        fclass += coords
    fclass /= len(faces)

    return fclass


def reconstruct(mean, basis, coords):
    res = np.matmul(coords, basis)
    res += mean
    return res
