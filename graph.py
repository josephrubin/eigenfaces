import matplotlib.pyplot as plt
import csv


__author__ = "Joseph, Fernando"


def graph():
    x = []
    y = []

    with open("vcount.txt") as f:
        data = csv.reader(f, delimiter=',')
        for row in data:
            x.append(int(row[0]))
            y.append(int(row[1]))

    plt.plot(x, y)

    plt.axis((0, 150, 0, 16))
    plt.yticks(range(0, 16))
    plt.xticks(list(range(0, 150, 15)) + [150])

    plt.xlabel('Eigenface Count')
    plt.ylabel('Number of Correct Classifications')

    plt.title("Correct Classifications vs Eigenface Count")

    # plt.show()
    plt.savefig("vcount.png")


if __name__ == "__main__":
    graph()
