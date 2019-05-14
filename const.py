from os import sep


__author__ = "Joseph, Fernando"


# Width of each image in px.
IMG_WIDTH = 200

# Height of each image in px.
IMG_HEIGHT = 235

# Pixel count for each image.
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT

# We will pick this many eigenfaces to use, chosen as
# the eigenfaces corresponding to the highest eigenvalues.
NUM_EIGENFACES = 150

# Which face from each individual will be reserved as
# the test face.
TEST_FACE_INDEX = 0

# The standard deviation of the Gaussian Blur filter
# applied to each test image before classification,
# or 0 for no blur.
# Empirically, best value is 9.
BLUR_AMOUNT = 9

# Location of cropped photos relative to source file directory.
ASSET_LOCATION_TEMPLATE = "asset" + sep + "crop_subject{0:0>2d}"

# If true, print out progress during computations,
# as well as final result during a call to main.run_trials.
VERBOSE_MODE = True
