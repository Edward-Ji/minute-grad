import csv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from util import BatchGenerator


def train_loop(
    X_train, y_train, X_test, y_test, epochs, batch_size, model, optimiser, loss_fn
):
    batches = BatchGenerator(X_train, y_train, batch_size=batch_size)

    train_loss_lst = []
    train_acc_lst = []

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_accuracy = 0
        model.train(True)
        for X_batch, y_batch in batches:
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimiser.optimise()

            train_loss += loss.val.item()
            train_accuracy += np.mean(np.argmax(logits.val, axis=1) == y_batch.val)

        train_loss /= len(batches)
        train_accuracy /= len(batches)

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_accuracy)

        if epoch % 10 == 0:
            model.train(False)
            test_logits = model(X_test)
            test_loss = loss_fn(test_logits, y_test).val.item()
            test_accuracy = np.mean(np.argmax(test_logits.val, axis=1) == y_test.val)
            tqdm.write(
                f"Epoch {epoch}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Train Accuracy: {train_accuracy:.2%}, "
                f"Test Loss: {test_loss:.4f},"
                f"Test Accuracy: {test_accuracy:.2%}"
            )

    return train_loss_lst, train_acc_lst, test_loss, test_accuracy


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
    # Plotting
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


def save_loss_accuracy(filename, labels, train_loss, train_acc, losses, accuracies):
    with open(f"{filename}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Model",
                "Training Loss",
                "Training Accuracy",
                "Test Loss",
                "Test Accuracy",
            ]
        )
        for label, t_loss, t_acc, loss, acc in zip(
            labels, train_loss, train_acc, losses, accuracies
        ):
            writer.writerow([label, t_loss, t_acc, loss, acc])
