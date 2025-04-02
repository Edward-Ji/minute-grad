from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class BatchGenerator:

    def __init__(self, *data, batch_size, shuffle=False):
        assert all(
            len(data[0]) == len(array) for array in data
        ), "All data arrays must have the same length"
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(self.data[0]), self.batch_size):
            end = min(start + self.batch_size, len(self.data[0]))
            batch_indices = indices[start:end]
            yield tuple(array[batch_indices] for array in self.data)


def unbroadcast(grad: NDArray, shape):
    # If grad has more dimensions than shape, sum over the extra dimensions.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # For dimensions where shape is 1, sum over that axis.
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def xavier_uniform(n_in: int, n_out: int):
    high = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low=-high, high=high, size=(n_in, n_out))


def xavier_normal(n_in: int, n_out: int):
    scale = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(scale=scale, size=(n_in, n_out))


def kaiming_uniform(n_in: int, n_out: int):
    high = np.sqrt(6 / n_in)
    return np.random.uniform(low=-high, high=high, size=(n_in, n_out))


def kaiming_normal(n_in: int, n_out: int):
    return np.random.normal(scale=np.sqrt(2 / n_in), size=(n_in, n_out))


def train_test_split(X: NDArray, y: NDArray, test_size: float):
    n_test = int(len(y) * test_size)
    indices = np.random.permutation(len(y))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def load_data(data_path: Path):
    X_train = np.load(data_path / "train_data.npy")
    y_train = np.load(data_path / "train_label.npy")
    X_test = np.load(data_path / "test_data.npy")
    y_test = np.load(data_path / "test_label.npy")
    return X_train, y_train, X_test, y_test
