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
    LeakyReLU,
    Linear,
    BatchNormalisation
)
from optimiser import AdamOptimiser, GradientDescentOptimiser
from tensor import Tensor
from util import kaiming_uniform, min_max_scale, standard_scale
from train_util import save_loss_accuracy, train_loop, plot_losses_and_accuracies

epochs = [10, 20, 50, 100]

batches = [1, 4, 8, 32, 64, 128]

normalisations = [min_max_scale, standard_scale]

learning_rates = [0.1, 0.01, 0.001, 0.0001]

weight_decays = [0.01, 0.001, 0.0001, 0.00001]

optimisers = [AdamOptimiser, GradientDescentOptimiser]


def main():
    # Define defaults
    default_epoch = 100
    default_batch_size = 32
    default_normalisation = standard_scale
    default_learning_rate = 0.001
    default_weight_decay = 0.001
    default_optimiser = AdamOptimiser

    # Define all sweeps
    sweep_configs = {
        # "epoch": epochs,
        # "batch": batches,
        "normalisation": normalisations
        # "lr": learning_rates,
        # "wd": weight_decays,
        # "optimiser": optimisers
    }

    for sweep_type, sweep_values in sweep_configs.items():
        all_training_loss_lst = []
        all_train_acc_lst = []

        all_test_loss = []
        all_test_accuracy = []

        model_labels = []

        for value in sweep_values:
            # Define the model
            model = Composite(
                [
                    Linear(128, 256, initialise=kaiming_uniform),
                    BatchNormalisation(256),
                    LeakyReLU(),
                    Linear(256, 10, initialise=kaiming_uniform),
                ]
            )

            # Start with defaults
            epoch = default_epoch
            batch_size = default_batch_size
            normalisation = default_normalisation
            learning_rate = default_learning_rate
            weight_decay = default_weight_decay
            optimiser_cls = default_optimiser

            # Override just the parameter we're sweeping
            if sweep_type == "epoch":
                epoch = value
                model_labels.append(f'{value} epochs')
            elif sweep_type == "batch":
                batch_size = value
                model_labels.append(f'batch size {value}')
            elif sweep_type == "normalisation":
                normalisation = value
                name_dict = {0: "Min-max", 1: "Standard scaling"}
                name = name_dict[normalisations.index(normalisation)]
                model_labels.append(f'{name} normalisation')
            elif sweep_type == "lr":
                learning_rate = value
                model_labels.append(f'{value} learning rate')
            elif sweep_type == "wd":
                weight_decay = value
                model_labels.append(f'{value} weight decay')
            elif sweep_type == "optimiser":
                optimiser_cls = value
                model_labels.append(f'{value.name} optimiser')
            
            # Load data
            X_train = Tensor(normalisation(np.load("../data/train_data.npy")))
            y_train = Tensor(np.load("../data/train_label.npy").squeeze())
            X_test = Tensor(normalisation(np.load("../data/test_data.npy")))
            y_test = Tensor(np.load("../data/test_label.npy").squeeze())

            optimiser = optimiser_cls(
                model.get_all_tensors(),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
            loss_fn = CrossEntropyLoss(label_smoothing=0.3)

            # Train
            train_loss_lst, train_acc_lst, test_loss, test_accuracy = train_loop(
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

            all_training_loss_lst.append(train_loss_lst)
            all_train_acc_lst.append(train_acc_lst)

            all_test_loss.append(test_loss)
            all_test_accuracy.append(test_accuracy)

        folder = f"./experiments/results/{sweep_type}"

        plot_losses_and_accuracies(folder + "/train", model_labels, all_training_loss_lst, all_train_acc_lst)

        save_loss_accuracy(
            folder + sweep_type,
            sweep_values,
            [x[-1] for x in all_training_loss_lst],
            [x[-1] for x in all_train_acc_lst],
            all_test_loss,
            all_test_accuracy,
        )


if __name__ == "__main__":
    main()
