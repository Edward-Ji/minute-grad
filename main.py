import numpy as np
from tqdm.auto import tqdm

from layer import Composite, CrossEntropyLoss, Dropout, ReLU, LeakyReLU, Linear, BatchNormalisation
from optimiser import AdamOptimiser
from tensor import Tensor
from util import BatchGenerator, xavier_uniform, standard_scale

'''
This file contains the main training loop and testing results for our obtained most optimal model.
See the other files in this repo for the implementation of the different modules.
'''

def main():
    # Load and normalise the data
    X_train = Tensor(standard_scale(np.load("./data/train_data.npy", allow_pickle=True)))
    y_train = Tensor(np.load("./data/train_label.npy").squeeze())
    X_test = Tensor(standard_scale(np.load("./data/test_data.npy")))
    y_test = Tensor(np.load("./data/test_label.npy").squeeze())

    epochs = 100
    batch_size = 64

    # Split the data into batches
    batches = BatchGenerator(X_train, y_train, batch_size=batch_size)

    # Create the model, optimser and loss function
    model = Composite(
        [
            Linear(128, 192),
            BatchNormalisation(192),
            LeakyReLU(),
            Linear(192, 10, initialise=xavier_uniform),
        ]
    )
    optimiser = AdamOptimiser(model.get_all_tensors(), weight_decay=0.001)
    loss_fn = CrossEntropyLoss(label_smoothing=0.3)

    # Loop through and train the model
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        train_accuracy = 0
        model.train(True)
        for X_batch, y_batch in batches:
            optimiser.zero_grad()
            # Compute output
            logits = model(X_batch)
            # Compute loss
            loss = loss_fn(logits, y_batch)
            # Backpropagate
            loss.backward()
            # Optimise the parameters
            optimiser.optimise()
            # Add up loss and accuracy values
            train_loss += loss.val.item()
            train_accuracy += np.mean(np.argmax(logits.val, axis=1) == y_batch.val)
        # Get the average loss and accuracy values
        train_loss /= len(batches)
        train_accuracy /= len(batches)
        # Evaluate on the test set to see test performance
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


if __name__ == "__main__":
    main()
