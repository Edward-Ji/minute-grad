import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from layer import Composite, CrossEntropyLoss, ReLU, Linear
from optimiser import AdamOptimiser
from tensor import Tensor
from util import kaiming_uniform
from train_util import train_loop, plot_losses_and_accuracies, save_loss_accuracy


models = [
    Composite(
        [
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
]

model_labels = [
    "1 Layer",
    "2 Layers",
    "3 Layers",
    "4 Layers",
    "5 Layers",
]


def main():
    all_training_loss_lst = []
    all_train_acc_lst = []

    all_test_loss = []
    all_test_accuracy = []

    for model in models:
        X_train = Tensor(np.load("./data/train_data.npy"))
        y_train = Tensor(np.load("./data/train_label.npy").squeeze())
        X_test = Tensor(np.load("./data/test_data.npy"))
        y_test = Tensor(np.load("./data/test_label.npy").squeeze())

        epochs = 50
        batch_size = 64

        optimiser = AdamOptimiser(model.get_all_tensors(), weight_decay=1e-5)
        loss_fn = CrossEntropyLoss(label_smoothing=0.3)

        train_loss_lst, train_acc_lst, test_loss, test_accuracy = train_loop(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            batch_size,
            model,
            optimiser,
            loss_fn,
        )

        all_training_loss_lst.append(train_loss_lst)
        all_train_acc_lst.append(train_acc_lst)

        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)

    folder = "./experiments/results/depth_test"

    plot_losses_and_accuracies(
        folder + "/train", model_labels, all_training_loss_lst, all_train_acc_lst
    )
    save_loss_accuracy(
        folder + "/test",
        model_labels,
        all_test_loss,
        all_test_accuracy,
    )


if __name__ == "__main__":
    main()
