import numpy as np
import pandas as pd
import torch
from torch import nn, optim

columns = ["unit_nr", "time_cycles", "setting_1", "setting_2", "setting_3"]
for i in range(1, 22):
    columns.append("s_" + str(i))


def read_data(file_name):
    data = pd.read_csv(file_name, sep=r"\s+|,", engine="python", header=None)
    data = data.dropna(axis=1, how="all")

    first_row = pd.to_numeric(data.iloc[0], errors="coerce")
    if first_row.isna().any():
        data = data.iloc[1:]

    data.columns = columns
    return data.apply(pd.to_numeric)


def get_rul_data(train_file, test_file, truth_file):
    train = read_data(train_file)
    test = read_data(test_file)
    truth = pd.read_csv(truth_file, header=None).iloc[:, -1]
    truth = pd.to_numeric(truth, errors="coerce").dropna().to_numpy()

    last_cycle = train.groupby("unit_nr")["time_cycles"].transform("max")
    train["RUL"] = last_cycle - train["time_cycles"]
    train["RUL"] = train["RUL"].clip(upper=125)

    features = columns[1:]
    last_test_rows = test.groupby("unit_nr").last().reset_index()

    x_train = train[features].to_numpy()
    y_train = train["RUL"].to_numpy()
    x_test = last_test_rows[features].to_numpy()
    y_test = np.minimum(truth, 125)

    return x_train, y_train, x_test, y_test


class PyTorchRULModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Replicating your [input -> 32 -> 1] architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def main():
    try:
        x_train, y_train, x_test, y_test = get_rul_data(
            "PM_train.csv", "PM_test.csv", "PM_truth.csv"
        )
    except FileNotFoundError:
        print("You need these three files in the project folder:")
        print("PM_train.csv, PM_test.csv and PM_truth.csv")
        return

    # --- Standardize Data ---
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0)
    x_std[x_std == 0] = 1
    x_train_scaled = (x_train - x_mean) / x_std
    x_test_scaled = (x_test - x_mean) / x_std

    y_mean, y_std = y_train.mean(), y_train.std()
    if y_std == 0:
        y_std = 1
    y_train_scaled = (y_train - y_mean) / y_std

    # --- Convert to PyTorch Tensors ---
    X_train_t = torch.FloatTensor(x_train_scaled)
    y_train_t = torch.FloatTensor(y_train_scaled).view(-1, 1)
    X_test_t = torch.FloatTensor(x_test_scaled)

    # --- Initialize Model, Loss, and Optimizer ---
    torch.manual_seed(42)
    model = PyTorchRULModel(x_train.shape[1])
    criterion = nn.MSELoss()
    # Using Adam optimizer as it is generally preferred in PyTorch over vanilla GD
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # --- Training Loop ---
    epochs = 120
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear old gradients
        outputs = model(X_train_t)  # Forward pass
        loss = criterion(outputs, y_train_t)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

    # --- Prediction & Evaluation ---
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t)
        # Reverse the target scaling
        predictions = preds_scaled.numpy().ravel() * y_std + y_mean

    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    r2 = 1 - np.sum((predictions - y_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

    print("Test RMSE:", round(rmse, 2), "cycles")
    print("Test R2:", round(r2 * 100, 1), "%")

    print("\nFirst 5 test examples:")
    for i in range(min(5, len(y_test))):
        print(
            "Test engine",
            i + 1,
            "predicted:",
            round(predictions[i], 1),
            "actual:",
            round(y_test[i], 1),
        )


if __name__ == "__main__":
    main()
