import matplotlib.pyplot as plt


def display(face):
    image = face.reshape(235, 200)
    plt.matshow(image, cmap='gray', vmin=-1000, vmax=1000)
    plt.show()
