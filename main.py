import numpy as np
from sklearn.datasets import make_blobs

from layer import Composite, Dropout, Linear, MeanSquaredError, ReLU, Sigmoid
from optimiser import AdamOptimiser
from tensor import Tensor
from util import train_test_split

from util import xavier_uniform


def main():
    """
    Binary classification using a simple neural network with dropout.
    """
    X, y = make_blobs(centers=2)
    y = y[:, None]
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    X_train = Tensor(X_train)
    y_train = Tensor(y_train)
    X_test = Tensor(X_test)
    y_test = Tensor(y_test)

    # linear model
    model = Composite(
        [
            Linear(2, 32),
            ReLU(),
            Dropout(0.1),
            Linear(32, 32),
            ReLU(),
            Dropout(0.1),
            Linear(32, 1, initialise=xavier_uniform),
            Sigmoid(),
        ]
    )
    optimiser = AdamOptimiser(model.get_all_tensors())
    loss_fn = MeanSquaredError()

    pred = model(X_test)
    accuracy = np.mean((pred.val > 0.5) == y_test.val)
    print(f"Accuracy: {accuracy:.2f}")
    for _ in range(10_000):
        optimiser.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimiser.optimise()
    pred = model(X_test)
    accuracy = np.mean((pred.val > 0.5) == y_test.val)
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
