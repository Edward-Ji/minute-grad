import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from layer import (
    BatchNormalisation,
    Composite,
    CrossEntropyLoss,
    LeakyReLU,
    ReLU,
    Linear,
    Dropout,
    Sigmoid,
)
from optimiser import AdamOptimiser
from tensor import Tensor
from util import kaiming_uniform, xavier_uniform, standard_scale
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
            Linear(128, 384, initialise=kaiming_uniform),
            ReLU(),
            Linear(384, 10, initialise=kaiming_uniform),
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
    "384 Neurons",
    "512 Neurons",
]


dropout_models = [
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.2),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.4),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.6),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.8),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
]

dropout_labels = [
    "No Dropout",
    "0.2 Dropout",
    "0.4 Dropout",
    "0.6 Dropout",
    "0.8 Dropout",
]

dropout_3l_models = [
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Linear(192, 192, initialise=kaiming_uniform),
            ReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.2),
            Linear(192, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.2),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.4),
            Linear(192, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.4),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.6),
            Linear(192, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.6),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.8),
            Linear(192, 192, initialise=kaiming_uniform),
            ReLU(),
            Dropout(0.8),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
]

dropout_3l_labels = [
    "No Dropout 3 Layers",
    "0.2 Dropout 3 Layers",
    "0.4 Dropout 3 Layers",
    "0.6 Dropout 3 Layers",
    "0.8 Dropout 3 Layers",
]

activation_models = [
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            Linear(192, 10, initialise=kaiming_uniform),
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
            Linear(128, 192, initialise=kaiming_uniform),
            LeakyReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(
                128, 192, initialise=xavier_uniform
            ),  # xavier uniform was created for sigmoid activation whereas Kaiming was developed for ReLU activation.
            Sigmoid(),
            Linear(192, 10, initialise=xavier_uniform),
        ]
    ),
]

activation_labels = [
    "No Activation",
    "ReLU",
    "Leaky ReLU",
    "Sigmoid",
]

batchnorm_models = [
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            BatchNormalisation(192),
            LeakyReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
    Composite(
        [
            Linear(128, 192, initialise=kaiming_uniform),
            LeakyReLU(),
            Linear(192, 10, initialise=kaiming_uniform),
        ]
    ),
]

batchnorm_labels = ["With Batch Normalisation", "Without Batch Normalisation"]


experiments = [
    [depth_models, depth_labels]
    [width_models, width_labels],
    [dropout_models, dropout_labels],
    [dropout_3l_models, dropout_3l_labels],
    [activation_models, activation_labels],
    [batchnorm_models, batchnorm_labels]
]

folders = [
    "./experiments/results/depth_test",
    "./experiments/results/width_test",
    "./experiments/results/dropout_test",
    "./experiments/results/dropout_3l_test",
    "./experiments/results/activation_test",
    "./experiments/results/batchnorm_test",
]


def run_experiment(folder, models, model_labels):
    all_training_loss_lst = []
    all_train_acc_lst = []

    all_test_loss = []
    all_test_accuracy = []

    training_times = []
    inference_times = []

    for model in models:
        # Load and standardise the data
        X_train = Tensor(standard_scale(np.load("../data/train_data.npy")))
        y_train = Tensor(np.load("../data/train_label.npy").squeeze())
        X_test = Tensor(standard_scale(np.load("../data/test_data.npy")))
        y_test = Tensor(np.load("../data/test_label.npy").squeeze())

        epochs = 100
        batch_size = 64

        # Set up the optimiser and the loss function
        optimiser = AdamOptimiser(model.get_all_tensors(), weight_decay=1e-5)
        loss_fn = CrossEntropyLoss(label_smoothing=0.3)

        # Conduct the training loop to obtain the model's statistics on the data
        train_loss_lst, train_acc_lst, test_loss, test_accuracy, training_time, inference_time = train_loop(
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

        # Store this model's results to be plotted
        training_times.append(training_time)
        inference_times.append(inference_time)

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
        [x[-1] for x in all_training_loss_lst],
        [x[-1] for x in all_train_acc_lst],
        all_test_loss,
        all_test_accuracy,
        training_times,
        inference_times
    )


def main():
    for i, (models, labels) in enumerate(experiments):
        run_experiment(folders[i], models, labels)


if __name__ == "__main__":
    main()
