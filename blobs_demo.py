import numpy as np
from sklearn.datasets import make_blobs

from layer import Composite, CrossEntropyLoss, Dropout, Linear, ReLU
from optimiser import AdamOptimiser
from tensor import Tensor
from util import BatchGenerator, train_test_split, xavier_uniform


def main():
    """
    Multi-class classification using a simple neural network with dropout.
    """
    X: np.ndarray
    y: np.ndarray
    X, y = make_blobs(centers=3)  # pyright: ignore[reportAssignmentType]
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
    X_train = Tensor(X_train)
    y_train = Tensor(y_train)
    X_test = Tensor(X_test)
    y_test = Tensor(y_test)

    batches_trian = BatchGenerator(X_train, y_train, batch_size=32)

    # linear model
    model = Composite(
        [
            Linear(2, 32),
            ReLU(),
            Dropout(0.1),
            Linear(32, 32),
            ReLU(),
            Dropout(0.1),
            Linear(32, 3, initialise=xavier_uniform)
        ]
    )
    optimiser = AdamOptimiser(model.get_all_tensors())
    loss_fn = CrossEntropyLoss()

    model.train(False)
    logits = model(X_test)
    accuracy = np.mean(np.argmax(logits.val, axis=1) == y_test.val)
    print(f"Before training test accuracy: {accuracy:.2f}")
    for _ in range(100):
        model.train(True)
        for X_batch, y_batch in batches_trian:
            optimiser.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimiser.optimise()

    model.train(False)
    logits = model(X_test)
    accuracy = np.mean(np.argmax(logits.val, axis=1) == y_test.val)
    print(f"After training test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
