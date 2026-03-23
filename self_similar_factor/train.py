from structure import *
import pickle
import numpy as np
import sys
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

argErrCode=3
if (len(sys.argv)!=3):
    print("wrong number of arguments")
    print("example: python ./data_directory num_epochs")
    exit(argErrCode)

data_dir=str(sys.argv[1])

num_epochs = int(sys.argv[2])
learning_rate = 0.5e-1
learning_rate_final=1e-3
weight_decay = 1e-5
decrease_over = 50

in_pkl_train_file=data_dir+"/train_dataset.pkl"
step_of_decrease = num_epochs // decrease_over
print(f"step_of_decrease={step_of_decrease}")
gamma = (learning_rate_final/learning_rate) ** (1/step_of_decrease)

print(f"gamma={gamma}")
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device="+str(device))

with open(in_pkl_train_file,"rb") as fptr:
    train_data =pickle.load(fptr)

X_train_tensor = torch.tensor(train_data['X_train'], dtype=torch.float32)
Y_train_tensor = torch.tensor(train_data['Y_train'], dtype=torch.float32)

# Instantiate the dataset
train_dataset = CustomDataset(X_train_tensor, Y_train_tensor)

# Create DataLoader for training
batch_size = 50 # Define batch size
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

input_dim=1
output_dim=1
num_layers=3 #layers 0,1,2,3,.. num_layers
# A=1
model=self_similar_model(1,1,num_layers).to(device)

# Optimizer, scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Define a step learning rate scheduler
# Reduce learning rate by a factor of gamma every step_size epochs
scheduler = StepLR(optimizer, step_size=decrease_over, gamma=gamma)

# Define a regularization weight (hyperparameter)
# You may need to tune this value depending on how large the reg_loss gets
lambda_reg = 1e-3

print("Starting training...")
t_start=datetime.now()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_reg = 0.0

    for batch_X, batch_Y in train_loader:
        # Move data to the correct device
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)

        # 1. Zero the gradients
        optimizer.zero_grad()

        # 2. Forward pass: get predictions and regularization loss
        predictions, reg_loss = model(batch_X)

        # 3. Compute Complex MSE Loss manually
        # torch.abs() on a complex tensor computes the magnitude: sqrt(Re^2 + Im^2)
        # Squaring it gives Re^2 + Im^2. Taking the mean gives the MSE.
        mse_loss = torch.mean(torch.abs(predictions - batch_Y.to(torch.cfloat))**2)

        # 4. Combine losses
        loss = mse_loss + lambda_reg * reg_loss

        # 5. Backward pass
        loss.backward()

        # 6. Update weights
        optimizer.step()

        # Track metrics for logging
        epoch_loss += loss.item()
        epoch_mse += mse_loss.item()
        epoch_reg += reg_loss.item()

    # Step the learning rate scheduler once per epoch
    scheduler.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:
        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        avg_reg = epoch_reg / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Total Loss: {avg_loss:.6f} | "
              f"MSE: {avg_mse:.6f} | "
              f"Reg: {avg_reg:.6f} | "
              f"LR: {current_lr:.6e}")

print("Training complete!")
t_end=datetime.now()
print("time: ",t_end-t_start)
# Optional: Save the trained model weights
torch.save(model.state_dict(), data_dir + "/model_weights.pth")