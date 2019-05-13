import collect
import matplotlib.pyplot as plt
import os
import imageio


__author__ = "Joseph, Fernando"


ASSET_LOCATION_TEMPLATE = "asset" + os.sep + "crop_subject{0:0>2d}"


def write_images(subject_number, images):
    directory = ASSET_LOCATION_TEMPLATE.format(subject_number)
    os.mkdir(directory)
    for i, image in enumerate(images):
        imageio.imwrite(
            directory + os.sep + str(i) + ".png",
            image)


def main():
    NUM = 15
    WRITE = True

    images = collect.get_images(NUM)
    images_cropped = [img[8:243, 75:275] for img in images]

    if not WRITE:
        for i in images_cropped:
            plt.matshow(i)
            plt.show()
    else:
        write_images(NUM, images_cropped)


main()
