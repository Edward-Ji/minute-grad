import csv
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from util import BatchGenerator


def train_loop(
    X_train, y_train, X_test, y_test, epochs, batch_size, model, optimiser, loss_fn
):
    # Define a training loop to be used by a model, which conducts both training and testing of the model
    batches = BatchGenerator(X_train, y_train, batch_size=batch_size)

    train_loss_lst = []
    train_acc_lst = []
    start_time = time.time()

    # Training loop
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_accuracy = 0
        model.train(True)
        # For each batch, produce a prediction and calculate the loss
        for X_batch, y_batch in batches:
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimiser.optimise()

            train_loss += loss.val.item()
            train_accuracy += np.mean(np.argmax(logits.val, axis=1) == y_batch.val)
        
        if optimiser.name == 'Adam':
            optimiser.iterations += 1
        
        # Calculate the training loss and accuracy

        train_loss /= len(batches)
        train_accuracy /= len(batches)

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_accuracy)

    # Note the time taken to train
    training_time = time.time() - start_time

    # After training is completed, test the model to see how well it performs
    start_time = time.time()
    model.train(False)
    test_logits = model(X_test)
    inference_time = time.time() - start_time
    test_loss = loss_fn(test_logits, y_test).val.item()
    test_accuracy = np.mean(np.argmax(test_logits.val, axis=1) == y_test.val)
    tqdm.write(
        f"Epoch {epoch}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2%}, "
        f"Test Loss: {test_loss:.4f},"
        f"Test Accuracy: {test_accuracy:.2%}"
    )

    # Reset the optimiser if necessary
    if optimiser.name == 'Adam':
        optimiser.iterations = 1

    return train_loss_lst, train_acc_lst, test_loss, test_accuracy, training_time, inference_time


def plot_losses_and_accuracies(
    filename_prefix, model_labels, all_training_losses, all_training_accuracies
):
    os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # --- Plot training losses ---
    for i, losses in enumerate(all_training_losses):
        label = model_labels[i] if i < len(model_labels) else f"Model {i}"
        ax1.plot(losses, label=label)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Epochs")
    ax1.legend()
    ax1.grid(True)

    # --- Plot training accuracies ---
    for i, accs in enumerate(all_training_accuracies):
        label = model_labels[i] if i < len(model_labels) else f"Model {i}"
        ax2.plot(accs, label=label)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy over Epochs")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}.png", dpi=300)
    plt.close()


def plot_losses(filename, model_labels, training_losses):
    # Plotting training lossses across the number of epochs trained for each of the different model types
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(training_losses):
        label = model_labels[i] if i < len(model_labels) else f"Model {i}"
        plt.plot(losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()


def save_loss_accuracy(filename, labels, train_loss, train_acc, losses, accuracies, training_time, inference_time):
    # Produce a CSV which contains the training and test losses and accuracies, as well as the training and inference times
    with open(f"{filename}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Model",
                "Training Loss",
                "Training Accuracy",
                "Test Loss",
                "Test Accuracy",
                "Training time (seconds)",
                "Inference time (seconds)"
            ]
        )
        for label, t_loss, t_acc, loss, acc, train_time, test_time in zip(
            labels, train_loss, train_acc, losses, accuracies, training_time, inference_time
        ):
            writer.writerow([label, t_loss, t_acc, loss, acc, train_time, test_time])
