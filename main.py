import numpy as np
from tqdm.auto import tqdm

from layer import Composite, CrossEntropyLoss, Dropout, LeakyReLU, Linear
from optimiser import AdamOptimiser
from tensor import Tensor
from util import BatchGenerator, xavier_uniform


def main():
    X_train = Tensor(np.load("./data/train_data.npy"))
    y_train = Tensor(np.load("./data/train_label.npy").squeeze())
    X_test = Tensor(np.load("./data/test_data.npy"))
    y_test = Tensor(np.load("./data/test_label.npy").squeeze())

    epochs = 100
    batch_size = 64

    batches = BatchGenerator(X_train, y_train, batch_size=batch_size)

    model = Composite(
        [
            Linear(128, 512),
            LeakyReLU(),
            Dropout(0.3),
            Linear(512, 256),
            LeakyReLU(),
            Dropout(0.3),
            Linear(256, 128),
            LeakyReLU(),
            Dropout(0.3),
            Linear(128, 10, initialise=xavier_uniform),
        ]
    )
    optimiser = AdamOptimiser(model.get_all_tensors(), weight_decay=1e-5)
    loss_fn = CrossEntropyLoss(label_smoothing=0.3)

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
