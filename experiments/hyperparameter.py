from itertools import product
import sys
import os

import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from layer import (
    Composite,
    CrossEntropyLoss,
    ReLU,
    Linear,
)
from optimiser import AdamOptimiser, GradientDescentOptimiser
from tensor import Tensor
from util import kaiming_uniform, min_max_scale, standard_scale
from train_util import train_loop

model = Composite(
    [
        Linear(128, 192, initialise=kaiming_uniform),
        ReLU(),
        Linear(192, 10, initialise=kaiming_uniform),
    ]
)

epochs = [50, 100, 150]

batches = [1, 4, 8, 32, 64, 128]

normalisations = [min_max_scale, standard_scale]

learning_rates = [0.1, 0.01, 0.001, 0.0001]

weight_decays = [0.01, 0.001, 0.0001, 0.00001]

optimisers = [AdamOptimiser, GradientDescentOptimiser]


def log_final_test_results(
    output_file,
    epoch,
    batch_size,
    normalisation_name,
    learning_rate,
    weight_decay,
    optimiser_name,
    test_loss,
    test_accuracy,
):
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header on first write
            writer.writerow(
                [
                    "Epochs",
                    "Batch Size",
                    "Normalisation",
                    "Learning Rate",
                    "Weight Decay",
                    "Optimiser",
                    "Test Loss",
                    "Test Accuracy",
                ]
            )

        writer.writerow(
            [
                epoch,
                batch_size,
                normalisation_name,
                learning_rate,
                weight_decay,
                optimiser_name,
                test_loss,
                test_accuracy,
            ]
        )


def main():
    print(AdamOptimiser.__name__)

    for (
        epoch,
        batch_size,
        normalisation,
        learning_rate,
        weight_decay,
        optimiser,
    ) in product(
        epochs, batches, normalisations, learning_rates, weight_decays, optimisers
    ):
        X_train = Tensor(normalisation(np.load("./data/train_data.npy")))
        y_train = Tensor(np.load("./data/train_label.npy").squeeze())
        X_test = Tensor(normalisation(np.load("./data/test_data.npy")))
        y_test = Tensor(np.load("./data/test_label.npy").squeeze())

        optimiser = optimiser(
            model.get_all_tensors(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        loss_fn = CrossEntropyLoss(label_smoothing=0.3)

        _, _, test_loss, test_accuracy = train_loop(
            X_train,
            y_train,
            X_test,
            y_test,
            epoch,
            batch_size,
            model,
            optimiser,
            loss_fn,
        )

        log_final_test_results(
            "./experiments/results/hyperparam_log.csv",
            epoch,
            batch_size,
            normalisation.__name__,
            learning_rate,
            weight_decay,
            optimiser.__name__,
            test_loss,
            test_accuracy,
        )


if __name__ == "__main__":
    main()
