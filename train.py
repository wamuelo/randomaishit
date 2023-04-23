import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from dataset import MyDataset
from model import MyModel

# Load config file
with open("config.json", "r") as f:
    config = json.load(f)

# Set random seed for reproducibility
torch.manual_seed(42)

# Initialize model and optimizer
model = MyModel(config["hidden_size"], len(MyDataset.CHARACTERS))
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()

# Load dataset
data_path = config["data_path"]
seq_length = config["seq_length"]
dataset = MyDataset(data_path, seq_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# Train the model
total_steps = 0
for epoch in range(config["num_epochs"]):
    for i, (inputs, targets) in enumerate(dataloader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs.view(-1, len(MyDataset.CHARACTERS)), targets.view(-1))

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        total_steps += 1

        # Save model at intervals
        if total_steps % config["save_interval"] == 0:
            torch.save(model.state_dict(), f"{config['model_path']}/G_{total_steps}")
            print(f"Model saved at {config['model_path']}/G_{total_steps}")

    # Save model after training on all files
    if epoch == config["num_epochs"] - 1:
        torch.save(model.state_dict(), f"{config['model_path']}/G_{total_steps}")
        print(f"Model saved at {config['model_path']}/G_{total_steps}")
