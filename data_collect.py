import main


__author__ = "Joseph, Fernando"


# This file was just used for collecting some data
# for the paper.

# Set VERBOSE_MODE to False in const.py before using this file,
# for your own sanity!


def vary_eigenfaces_count():
    with open("vcount.txt", "w") as f:
        for i in range(1, 150 + 1):
            result, _ = main.run_trials(0, i)
            print(i, result)
            f.write(str(i) + "," + str(result) + "\n")


def vary_test_face():
    with open("vface.txt", "w") as f:
        for i in range(0, 10):
            result, _ = main.run_trials(i, 150)
            print(i, result)
            f.write(str(i) + "," + str(result) + "\n")


if __name__ == "__main__":
    # vary_test_face()
    vary_eigenfaces_count()
