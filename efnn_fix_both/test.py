import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from structure import efnn, CustomDataset
from torch.utils.data import DataLoader

def main():
    argErrCode = 3
    if len(sys.argv) != 3:
        print("Wrong number of arguments.")
        print("Example: python test.py ./bd num_neurons")
        sys.exit(argErrCode)

    data_dir = str(sys.argv[1])
    num_neurons = int(sys.argv[2])

    # Hyperparameters and constants (must match train.py and gen_bd.py)
    a = -1
    b = 1.0
    fa = 0
    fb = 2
    eps = 0.05
    num_layers = 3

    in_pkl_test_file = f"{data_dir}/test_dataset_eps{eps}.pkl"
    model_weights_file = f"{data_dir}/model_weights_eps{eps}.pth"

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the testing data
    try:
        with open(in_pkl_test_file, "rb") as fptr:
            test_data = pickle.load(fptr)
    except FileNotFoundError:
        print(f"Error: Test data file {in_pkl_test_file} not found.")
        sys.exit(1)

    X_test_np = test_data['X_test']
    Y_test_np = test_data['Y_test']

    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test_np, dtype=torch.float32)

    test_dataset = CustomDataset(X_test_tensor, Y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=500, shuffle=False)

    # 2. Initialize Model and load weights
    model = efnn(1, num_layers, num_neurons, a, b, fa, fb, eps).to(device)

    try:
        model.load_state_dict(torch.load(model_weights_file, map_location=device))
        print(f"Successfully loaded model weights from {model_weights_file}")
    except FileNotFoundError:
        print(f"Error: Model weights file {model_weights_file} not found.")
        sys.exit(1)

    model.eval()

    # 3. Evaluate the model
    total_mse = 0.0
    all_predictions = []
    all_targets = []
    all_inputs = []

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)

            predictions = model(batch_X)

            # MSE calculation (matching the logic in train.py)
            mse_loss = torch.mean(torch.abs(predictions - batch_Y.to(torch.cfloat))**2)
            total_mse += mse_loss.item() * batch_X.size(0)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_Y.cpu().numpy())
            all_inputs.append(batch_X.cpu().numpy())

    avg_mse = total_mse / len(test_dataset)
    print(f"Test Mean Squared Error (MSE): {avg_mse:.6e}")

    # 4. Plot the results
    # Concatenate all batches for plotting
    X_plot = np.concatenate(all_inputs).flatten()
    Y_pred_plot = np.concatenate(all_predictions).flatten()
    Y_exact_plot = np.concatenate(all_targets).flatten()

    # Sort the values by X so the line plot connects correctly
    sort_idx = np.argsort(X_plot)
    X_plot = X_plot[sort_idx]
    Y_pred_plot = Y_pred_plot[sort_idx]
    Y_exact_plot = Y_exact_plot[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(X_plot, Y_exact_plot, label="Exact Solution", color="blue", linewidth=2)
    plt.plot(X_plot, Y_pred_plot, label="EFNN Prediction", color="red", linestyle="--", linewidth=2)

    plt.title(f"EFNN Prediction vs Exact Solution (eps={eps})")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)

    plot_file = f"{data_dir}/results_plot_eps{eps}.png"
    plt.savefig(plot_file)
    print(f"Plot saved to {plot_file}")

    # plt.show()

if __name__ == "__main__":
    main()