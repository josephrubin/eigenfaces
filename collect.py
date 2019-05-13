import imageio
import os


ASSET_LOCATION_TEMPLATE = "asset" + os.sep + "crop_subject{0:0>2d}"


def get_images(subject_number):
    directory = ASSET_LOCATION_TEMPLATE.format(subject_number)
    images = []
    for file_name in os.listdir(directory):
        images.append(imageio.imread(directory + os.sep + file_name))

    return images


def get_images_all():
    images = []
    for i in range(1, 15 + 1):
        images.extend(get_images(i))
    return images
