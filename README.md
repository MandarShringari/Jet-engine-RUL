# Simple RUL prediction

This project predicts an engine's remaining useful life (RUL) from sensor data.
It uses a small custom neural network with one hidden layer and plain gradient
descent—no Adam optimiser—so there are fewer settings to understand.

Place these three files in the project directory:

- `PM_train.csv` — training sensor readings
- `PM_test.csv` — test sensor readings
- `PM_truth.csv` — one true RUL value for each test engine

Put the three data files in the project folder, then run:

```bash
uv run cs-ai-lab
```

The script reports RMSE in cycles and R². R² is the accuracy-like score; around
80% is a reasonable goal, but the exact score depends on the supplied data.
