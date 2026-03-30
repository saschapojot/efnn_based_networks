import numpy as np
from pathlib import Path
import pickle
from scipy.special import airy

def mirror_airy(x):
    """
    x>0
    """
    y = -x
    # scipy.special.airy returns 4 values: Ai, Ai', Bi, Bi'
    # We unpack them and return only Ai (Airy function of the first kind)
    ai, aip, bi, bip = airy(y)
    return 5*ai+20


def generate_X_train(num_train, x_min=0.0, x_max=10.0):
    """
    Generates training input data in the range [x_min, x_max] with boundary points included.
    """
    X_random = np.random.uniform(x_min, x_max, size=(num_train - 2, 1))
    # Explicitly create the boundary points
    X_boundaries = np.array([[x_min], [x_max]])
    # Combine the random points and the boundary points
    X_train = np.vstack((X_random, X_boundaries))
    # Shuffle the training data so the boundaries aren't always at the very end
    np.random.shuffle(X_train)

    return X_train

def generate_X_test(num_test, x_min=0.0, x_max=10.0):
    """
    Generates testing input data evenly spaced in the range [x_min, x_max].
    """
    # Using linspace for testing data to plot a smooth curve later
    X_test = np.linspace(x_min, x_max, num_test).reshape(-1, 1)
    return X_test


out_data_dir = "./Airy_vals/"
Path(out_data_dir).mkdir(exist_ok=True, parents=True)
num_train = 500
num_test = 500
x_min = 0
x_max = 10
x_test_max = 15

# 1. Generate X_train and Y_train separately
X_train = generate_X_train(num_train, x_min, x_max)
Y_train = mirror_airy(X_train)
print("Preview of Y_train:\n", Y_train[:5]) # Printed just the first 5 for cleaner output

# 2. Generate X_test and Y_test separately
X_test = generate_X_test(num_test, x_min, x_test_max)
Y_test = mirror_airy(X_test)

# This will now work correctly because Y_train and Y_test are NumPy arrays
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
train_save_path = out_data_dir + "/train_dataset.pkl"
with open(train_save_path, 'wb') as f:
    pickle.dump(train_dict, f)
print(f"Training data successfully saved to {train_save_path}")


# Save the testing data to a .pkl file
test_save_path = out_data_dir + "/test_dataset.pkl"
with open(test_save_path, 'wb') as f:
    pickle.dump(test_dict, f)
print(f"Testing data successfully saved to {test_save_path}")