import imageio
import os


ASSET_LOCATION_TEMPLATE = "asset" + os.sep + "subject{0:0>2d}"


def get_images(subject_number):
    directory = ASSET_LOCATION_TEMPLATE.format(subject_number)
    print(directory)
    images = []
    for file_name in os.listdir(directory):
        images.append(imageio.imread(directory + os.sep + file_name))

    return images