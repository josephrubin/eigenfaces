import matplotlib.pyplot as plt


def display(face, min_=0, max_=255):
    """Display a grayscale version of the given face vector.

    The levels are set such that white is min_, and
    black is max_.
    """
    image = face.reshape(235, 200)
    plt.matshow(image, cmap='gray', vmin=min_, vmax=max_)
    plt.show()
