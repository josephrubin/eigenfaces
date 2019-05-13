import imageio
import os

from const import *


__author__ = "Joseph, Fernando"


CACHE = {}


def get_images(subject_number):
    if subject_number in CACHE:
        return CACHE[subject_number]

    directory = ASSET_LOCATION_TEMPLATE.format(subject_number)
    images = []
    for file_name in os.listdir(directory):
        images.append(imageio.imread(directory + os.sep + file_name))
    CACHE[subject_number] = images

    return images


def get_images_all():
    images = []
    for i in range(1, 15 + 1):
        images.extend(get_images(i))
    return images
