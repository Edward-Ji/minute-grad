# Minute Grad

A CPU implementation of an automatic differentiation framework for deep
learning.

## Run Locally

The automatic differentiation framework only depends on [NumPy] for computation.
For experiments, there are optional dependencies matplotlib, tqdm, and wandb for
visualisation, progress meter and tracking results respectively.

The project uses uv for managing dependencies, so you need to [install uv]
first. Then, run `uv sync --no-dev` to update your local environment. To run the
`experiments/all_experiments.py` script for example, simply run `uv run python
experiments/all_experiments.py`.

## Project Structure

```plaintext
.
├── data                             # Provided dataset in NumPy format
│   └── ...
├── experiments                      # Experimental scripts and utilities
│   ├── results                      # Directory containing experiment results (metrics & plots)
│   │   └── ...
│   ├── all_experiments.py           # Runs all experiments
│   ├── hyperparameter.py            # Hyperparameter tuning experiments
│   └── train_util.py                # Helper functions for training routines
├── layer.py                         # Implementation of network layers
├── main.py                          # An example training script on the provided dataset
├── optimiser.py                     # Definitions of optimizer algorithms (e.g., Adam)
├── tensor.py                        # Definition of tensor operations and structures
├── util.py                          # Miscellaneous helper functions
├── README.md                        # You are here.
├── pyproject.toml                   # Project configuration and dependency management
└── uv.lock                          # Dependency version lock file
```

[NumPy]: https://numpy.org/
[install uv]: https://docs.astral.sh/uv/getting-started/installation/
