import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from layer import Composite, CrossEntropyLoss, ReLU, Linear
from optimiser import AdamOptimiser
from tensor import Tensor
from util import kaiming_uniform, min_max_scale
from train_util import train_loop, plot_losses_and_accuracies, save_loss_accuracy


depth_models = [
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

depth_labels = [
    "1 Layer",
    "2 Layers",
    "3 Layers",
    "4 Layers",
    "5 Layers",
]

width_models = [
    Composite(
        [
            Linear(128, 128, initialise=kaiming_uniform),
            ReLU(),
            Linear(128, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 256, initialise=kaiming_uniform),
            ReLU(),
            Linear(256, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 512, initialise=kaiming_uniform),
            ReLU(),
            Linear(512, 10, initialise=kaiming_uniform),
        ]
    ),
]

width_labels = [
    "128 Neurons",
    "192 Neurons",
    "256 Neurons",
    "512 Neurons",
]


experiments = [
    [depth_models, depth_labels],
    [width_models, width_labels],
]

folders = ["./experiments/results/depth_test", "./experiments/results/width_test"]


def run_experiment(folder, models, model_labels):
    all_training_loss_lst = []
    all_train_acc_lst = []

    all_test_loss = []
    all_test_accuracy = []

    for model in models:
        X_train = min_max_scale(Tensor(np.load("./data/train_data.npy")))
        y_train = min_max_scale(Tensor(np.load("./data/train_label.npy").squeeze()))
        X_test = min_max_scale(Tensor(np.load("./data/test_data.npy")))
        y_test = min_max_scale(Tensor(np.load("./data/test_label.npy").squeeze()))

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

    plot_losses_and_accuracies(
        folder + "/train", model_labels, all_training_loss_lst, all_train_acc_lst
    )
    save_loss_accuracy(
        folder + "/test",
        model_labels,
        all_test_loss,
        all_test_accuracy,
    )


def main():
    for i, (models, labels) in enumerate(experiments):
        run_experiment(folders[i], models, labels)


if __name__ == "__main__":
    main()
