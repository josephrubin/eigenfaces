import matplotlib.pyplot as plt


def display(face):
    image = face.reshape(235, 200)
    plt.matshow(image)
    plt.show()