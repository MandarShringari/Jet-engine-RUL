# Jet Engine RUL Prediction

This project predicts a jet engine's Remaining Useful Life (RUL) from sensor data. It relies on a custom neural network built from scratch to demonstrate foundational machine learning concepts.

## Overview
Instead of relying on advanced optimizers or deep architectures, this project intentionally uses a simplified approach so there are fewer settings to understand:
* **Architecture:** A small custom neural network with a single hidden layer.
* **Optimization:** Plain gradient descent (no Adam optimizer).
* **Environment:** Built and managed using `uv`.

## Dataset
The repository includes the required dataset files directly:
* `PM_train.csv`: Training data containing engine sensor readings.
* `PM_test.csv`: Testing data.
* `PM_truth.csv`: The ground truth RUL values for the test set.

## Getting Started

### Prerequisites
Make sure you have [uv](https://github.com/astral-sh/uv) installed on your machine.

### Installation
Clone this repository and sync the dependencies using `uv`:

```bash
git clone [https://github.com/MandarShringari/Jet-engine-RUL.git](https://github.com/MandarShringari/Jet-engine-RUL.git)
cd Jet-engine-RUL
uv sync
