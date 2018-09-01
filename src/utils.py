import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_csv(file):
    with open(file, "r") as csvFile:
        data = []
        for row in csv.reader(csvFile, delimiter=","):
            data.append(np.array([row[1:]]).astype(np.float32))
        return data


def read_weights(file, input_dimentions):
    weights = []
    data = read_csv(file)
    idx = 0
    for dim in input_dimentions:
        W = np.reshape(data[idx:idx + dim], (dim, data[idx].shape[1]))
        weights.append(W)
        idx += dim
    return weights


def load_data(directory, scope="train"):
    X = np.loadtxt(f"{directory}/x_{scope}.csv", delimiter=",")
    y = np.loadtxt(f"{directory}/y_{scope}.csv", delimiter=",", dtype=int)
    assert X.shape[0] == y.shape[0]
    return X, y


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_epoch(directory, name, lr, batch_size, iterations, train_loss, train_acc, val_loss, val_acc, elapsed_time):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.8, 8), sharex="col", sharey="row")
    fig.suptitle(f"Neural Network: {name}", fontsize=18)
    rate_patch = mpatches.Patch(color="white")
    batch_patch = mpatches.Patch(color="white")
    elps_patch = mpatches.Patch(color="white")

    fig.legend((rate_patch, batch_patch, elps_patch),
               (f"Rate = {lr}",
                f"Batch = {batch_size}",
                "Time: {0:.2f}s".format(elapsed_time)),
               "upper right")

    ax1.plot(range(iterations), train_loss, color="b")
    ax1.set_title("Training cost w.r.t iterations")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True)

    ax2.plot(range(iterations), val_loss, color="r")
    ax2.set_title("Testing cost w.r.t iterations")
    ax2.grid(True)

    ax3.plot(range(iterations), train_acc, color="b")
    ax3.set_title("Training accuracy(%) w.r.t iterations")
    ax3.set_ylim([0, 1])
    vals = ax3.get_yticks()
    ax3.set_yticklabels(["{:3.0f}%".format(x * 100) for x in vals])
    ax3.set_xlabel("Iteration")
    ax3.set_xlim([0, iterations])
    ax3.set_ylabel("Accuracy")
    ax3.grid(True)

    ax4.plot(range(iterations), val_acc, color="r")
    ax4.set_title("Testing accuracy(%) w.r.t iterations")
    ax4.set_xlim([0, iterations])
    ax4.set_xlabel("Iteration")
    ax4.grid(True)

    plt.savefig(f"{directory}/img-{name}-B{batch_size}-R{lr}-I{iterations}.png")
