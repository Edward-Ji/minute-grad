from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class BatchGenerator:
    """
    A simple batch generator for iterating over data in batches. It expects any number
    of arrays as input, all of the same length. It yields batches of data in the form of
    tuples.
    """

    def __init__(self, *data, batch_size, shuffle=False):
        assert all(len(data[0]) == len(array) for array in data), (
            "All data arrays must have the same length"
        )
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.data[0]) / self.batch_size))

    def __iter__(self):
        indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(self.data[0]), self.batch_size):
            end = min(start + self.batch_size, len(self.data[0]))
            batch_indices = indices[start:end]
            yield tuple(array[batch_indices] for array in self.data)


def unbroadcast(grad: NDArray, shape):
    """
    A utillity function to unbroadcast a gradient tensor to match the shape of the
    corresponding parameter tensor. This is useful when the gradient tensor has been
    broadcasted to a larger shape during backpropagation implicitly by NumPy.
    """
    # If grad has more dimensions than shape, sum over the extra dimensions.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # For dimensions where shape is 1, sum over that axis.
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def xavier_uniform(n_in: int, n_out: int):
    """
    Xavier uniform initialization. This method is used to initialize the weights of
    neural networks. It is designed to keep the variance of the activations across
    layers approximately the same. The weights are drawn from a uniform distribution
    within the range [-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out))].
    """
    high = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(low=-high, high=high, size=(n_in, n_out))


def xavier_normal(n_in: int, n_out: int):
    """
    Xavier normal initialization. This method is used to initialize the weights of
    neural networks. It is designed to keep the variance of the activations across
    layers approximately the same. The weights are drawn from a normal distribution
    with mean 0 and standard deviation sqrt(2 / (n_in + n_out)).
    """
    scale = np.sqrt(2 / (n_in + n_out))
    return np.random.normal(scale=scale, size=(n_in, n_out))


def kaiming_uniform(n_in: int, n_out: int):
    """
    Kaiming uniform initialization. This method is used to initialize the weights of
    neural networks. It is designed to keep the variance of the activations across
    layers approximately the same. The weights are drawn from a uniform distribution
    within the range [-sqrt(6 / n_in), sqrt(6 / n_in)].
    """
    high = np.sqrt(6 / n_in)
    return np.random.uniform(low=-high, high=high, size=(n_in, n_out))


def kaiming_normal(n_in: int, n_out: int):
    """
    Kaiming normal initialization. This method is used to initialize the weights of
    neural networks. It is designed to keep the variance of the activations across
    layers approximately the same. The weights are drawn from a normal distribution
    with mean 0 and standard deviation sqrt(2 / n_in).
    """
    return np.random.normal(scale=np.sqrt(2 / n_in), size=(n_in, n_out))


def train_test_split(X: NDArray, y: NDArray, test_size: float):
    """
    Splits the data into training and testing sets. The test size is a fraction of the
    total data.
    """
    n_test = int(len(y) * test_size)
    indices = np.random.permutation(len(y))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def min_max_scale(X: NDArray, min_val: float = 0.0, max_val: float = 1.0):
    """
    Scales the data to a specified range using min-max scaling. The formula used is:
    X_scaled = min_val + (X - min_X) * (max_val - min_val) / (max_X - min_X)
    where min_X and max_X are the minimum and maximum values of the data along each
    feature dimension.
    """
    min_X = X.min(axis=0)
    max_X = X.max(axis=0)
    scale = (max_val - min_val) / (max_X - min_X)
    X_scaled = min_val + scale * (X - min_X)
    return X_scaled


def standard_scale(X: NDArray, mean: float = 0.0, std: float = 1.0):
    """
    Scales the data to have a specified mean and standard deviation. The formula is:
    X_scaled = mean + std * (X - mean_X) / std_X
    where mean_X and std_X are the mean and standard deviation of the data along each
    feature dimension.
    """
    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    X_scaled = mean + std * (X - mean_X) / std_X
    return X_scaled


def load_data(data_path: Path):
    """
    Loads the training and testing data from the specified path.
    """
    X_train = np.load(data_path / "train_data.npy")
    y_train = np.load(data_path / "train_label.npy")
    X_test = np.load(data_path / "test_data.npy")
    y_test = np.load(data_path / "test_label.npy")
    return X_train, y_train, X_test, y_test
