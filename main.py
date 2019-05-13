import numpy as np
import collect
import matplotlib.pyplot as plt

def main():
    #crop_data = open("crop_data", "r")

    images = collect.get_images(1)

    images_cropped = [img[5:240, 90:290] for img in images]

    images_vec = [img.flatten() for img in images_cropped]

    mean = np.zeros(235 * 200).flatten()
    for img in images_vec:
        mean = np.add(mean, img)

    mean /= len(images_vec)
    print(mean)

    for img in images_vec:
        i = img - mean

    mean.resize(235, 200)
    print(mean)
    plt.matshow(mean)
    plt.show()



main()