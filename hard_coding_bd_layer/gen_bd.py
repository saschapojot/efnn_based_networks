import numpy as np
from pathlib import Path
import pickle
def exact_solution(x, a, b, fa, fb, eps):
    """
    Computes the analytical solution based on the provided PDF.
    Handles complex roots if 1 - 4*eps < 0.
    Reformulated to avoid exponential overflow for small eps.
    """
    # Use complex numbers to handle negative values under the square root
    discriminant = 1.0 - 4.0 * eps + 0j

    # Calculate lambda_1 and lambda_2
    lambda1 = (-1.0 + np.sqrt(discriminant)) / (2.0 * eps)
    lambda2 = (-1.0 - np.sqrt(discriminant)) / (2.0 * eps)

    # To avoid overflow, we reformulate the solution as:
    # y(x) = C1 * exp(lambda1 * (x - b)) + C2 * exp(lambda2 * (x - a))
    # This ensures the arguments to exp() are bounded or large negative (which safely go to 0).

    E1 = np.exp(lambda1 * (a - b))
    E2 = np.exp(lambda2 * (b - a))

    denom = E1 * E2 - 1.0

    # Solve for C1 and C2 using the boundary conditions
    C1 = (fa * E2 - fb) / denom
    C2 = (fb * E1 - fa) / denom

    # Calculate exact y(x)
    y = C1 * np.exp(lambda1 * (x - b)) + C2 * np.exp(lambda2 * (x - a))

    # The physical solution is real, so we take the real part
    return np.real(y)


def generate_X_train(num_train, x_min, x_max):
    X_random = np.random.uniform(x_min, x_max, size=(num_train - 2, 1))

    # Explicitly create the boundary points
    X_boundaries = np.array([[x_min], [x_max]])

    # Combine the random points and the boundary points
    X_train = np.vstack((X_random, X_boundaries))

    # Shuffle the training data so the boundaries aren't always at the very end
    np.random.shuffle(X_train)

    return X_train

def generate_X_test(num_test, x_min, x_max):
    """
    Generates testing input data evenly spaced in the range [x_min, x_max].
    """
    # Using linspace for testing data to plot a smooth curve later
    X_test = np.linspace(x_min, x_max, num_test).reshape(-1, 1)
    return X_test

a = -1
b = 1.0
fa = 0         # y(0) = 0
fb = 2         # y(1) = 2
eps = 1e-1

out_data_dir="./bd/"
Path(out_data_dir).mkdir(exist_ok=True,parents=True)

num_train=1000
num_test=5000
x_min=-1
x_max=1
x_test_max=1

# 1. Generate X_train and Y_train separately
X_train = generate_X_train(num_train, x_min, x_max)
Y_train = exact_solution(X_train,a=a,b=b,fa=fa,fb=fb,eps=eps)

# 2. Generate X_test and Y_test separately
X_test = generate_X_test(num_test, x_min, x_test_max)
Y_test =exact_solution(X_test,a=a,b=b,fa=fa,fb=fb,eps=eps)

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Verify boundaries are in X_train
print(f"Minimum value in X_train: {np.min(X_train)}")
print(f"Maximum value in X_train: {np.max(X_train)}")

# Package the training data into a dictionary
train_dict = {
    'X_train': X_train,
    'Y_train': Y_train
}

# Package the testing data into a dictionary
test_dict = {
    'X_test': X_test,
    'Y_test': Y_test
}

# Save the training data to a .pkl file
train_save_path =out_data_dir+ f"/train_dataset_eps{eps}.pkl"
with open(train_save_path, 'wb') as f:
    pickle.dump(train_dict, f)
print(f"Training data successfully saved to {train_save_path}")


# Save the testing data to a .pkl file
test_save_path = out_data_dir+f"/test_dataset_eps{eps}.pkl"
with open(test_save_path, 'wb') as f:
    pickle.dump(test_dict, f)
print(f"Testing data successfully saved to {test_save_path}")
