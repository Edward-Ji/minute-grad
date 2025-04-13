import sys
import os
import time

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
from optimiser import GradientDescentOptimiser, AdamOptimiser
from tensor import Tensor
from util import kaiming_uniform, min_max_scale, standard_scale
from train_util import save_loss_accuracy, train_loop, plot_losses_and_accuracies

# Optimal model for reference
optimal_model = Composite(
    [
        Linear(128, 192, initialise=kaiming_uniform),
        BatchNormalisation(192),
        LeakyReLU(),
        Linear(192, 10, initialise=kaiming_uniform),
    ]
)

def main():
    # Optimal hyperparams
    epochs = 100
    batch_size = 64
    normalisation = standard_scale
    learning_rate = 0.001
    weight_decay = 0.001
    optimiser = AdamOptimiser

    # Define all sweeps
    sweep_configs = {
        "Optimal model": optimal_model,
        "No hidden layer": Composite(
            [
                Linear(128, 10, initialise=kaiming_uniform),
            ]
        ),
        "No activation": Composite(
            [
                Linear(128, 192, initialise=kaiming_uniform),
                BatchNormalisation(192),
                Linear(192, 10, initialise=kaiming_uniform),
            ]
        ),
        "No batch normalisation": Composite(
            [
                Linear(128, 192, initialise=kaiming_uniform),
                LeakyReLU(),
                Linear(192, 10, initialise=kaiming_uniform),
            ]
        )
    }

    all_training_loss_lst = []
    all_train_acc_lst = []

    all_test_loss = []
    all_test_accuracy = []

    training_times = []
    inference_times = []

    model_labels = list(sweep_configs.keys())

    for sweep_type, model in sweep_configs.items():
        # Load data
        X_train = Tensor(normalisation(np.load("../data/train_data.npy")))
        y_train = Tensor(np.load("../data/train_label.npy").squeeze())
        X_test = Tensor(normalisation(np.load("../data/test_data.npy")))
        y_test = Tensor(np.load("../data/test_label.npy").squeeze())

        optimiser_current = optimiser(
            model.get_all_tensors(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        loss_fn = CrossEntropyLoss(label_smoothing=0.3)

        # Train
        train_loss_lst, train_acc_lst, test_loss, test_accuracy, training_time, inference_time = train_loop(
            X_train,
            y_train,
            X_test,
            y_test,
            epochs,
            batch_size,
            model,
            optimiser_current,
            loss_fn,
        )

        training_times.append(training_time)
        inference_times.append(inference_time)

        all_training_loss_lst.append(train_loss_lst)
        all_train_acc_lst.append(train_acc_lst)

        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)

    folder = f"./experiments/results/ablation_study"

    plot_losses_and_accuracies(folder + "/train", model_labels, all_training_loss_lst, all_train_acc_lst)

    save_loss_accuracy(
        folder,
        model_labels,
        [x[-1] for x in all_training_loss_lst],
        [x[-1] for x in all_train_acc_lst],
        all_test_loss,
        all_test_accuracy,
        training_times,
        inference_times
    )

if __name__ == "__main__":
    main()