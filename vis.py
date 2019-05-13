import matplotlib.pyplot as plt
import imageio
import numpy as np

from const import *


__author__ = "Joseph, Fernando"


def display(face, min_=0, max_=255):
    """Display a grayscale version of the given face vector.

    The levels are set such that white is min_, and
    black is max_.
    """
    image = face.reshape(IMG_HEIGHT, IMG_WIDTH)
    plt.matshow(image, cmap='gray', vmin=min_, vmax=max_)
    plt.show()


def write(fname, face, min_=0, max_=255):
    """Normalize face to be within 0, 255,
    then write the image to the disk. For research
    purposes only - not used in the main program.
    """
    image = face.reshape(IMG_HEIGHT, IMG_WIDTH)
    res = plt.matshow(image, cmap='gray', vmin=min_, vmax=max_)
    res.axes.get_xaxis().set_visible(False)
    res.axes.get_yaxis().set_visible(False)
    plt.axis("off")
    plt.savefig(fname, bbox_inches="tight")
