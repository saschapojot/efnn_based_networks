import numpy as np
from pathlib import Path
import pickle



def func_exact(x):
    part0=(1+2*x)**(3/2)

    part1=(1+x)**(1/2)

    part2=(1+0.5*x)**(1/3)

    part3=(1+0.1*x)**(1/4)

    return part0*part1*part2*part3


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





out_data_dir="./exact_func/"
Path(out_data_dir).mkdir(exist_ok=True,parents=True)

num_train=500
num_test=500
x_min=0
x_max=20
x_test_max=25
# 1. Generate X_train and Y_train separately
X_train = generate_X_train(num_train, x_min, x_max)
Y_train = func_exact(X_train)
# 2. Generate X_test and Y_test separately
X_test = generate_X_test(num_test, x_min, x_test_max)
Y_test = func_exact(X_test)

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
train_save_path =out_data_dir+ "/train_dataset.pkl"
with open(train_save_path, 'wb') as f:
    pickle.dump(train_dict, f)
print(f"Training data successfully saved to {train_save_path}")


# Save the testing data to a .pkl file
test_save_path = out_data_dir+"/test_dataset.pkl"
with open(test_save_path, 'wb') as f:
    pickle.dump(test_dict, f)
print(f"Testing data successfully saved to {test_save_path}")
