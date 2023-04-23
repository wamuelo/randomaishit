import torch
import torch.nn as nn
import json
from dataset import MyDataset
from model import MyModel

with open('config.json', 'r') as f:
    config = json.load(f)

data_path = config['data_path']
model_path = config['model_path']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
hidden_size = config['hidden_size']
seq_length = config['seq_length']
save_interval = config['save_interval']

# Load dataset
dataset = MyDataset(data_path, seq_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define model
vocab_size = len(dataset.char2idx)
model = MyModel(vocab_size, hidden_size)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_steps = 0
hidden = None

for epoch in range(num_epochs):
    total_loss = 0
    for i, data in enumerate(dataloader):
        inputs = data[0].to(device)
        targets = data[1].to(device)

        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {total_loss/100:.4f}')
            total_loss = 0

        steps = (epoch+1)*len(dataset)
        if total_steps % save_interval == 0:
            torch.save(model.state_dict(), f'{model_path}/G_{steps}.pt')
            print(f"Model saved at {model_path}/G_{steps}.pt")

    #torch.save(model.state_dict(), f'{model_path}/model_{epoch+1}.pt')

# Save final model
steps = (epoch+1)*len(dataset)
torch.save(model.state_dict(), f'{model_path}/G_{steps}.pt')
print(f"Model saved at {model_path}/G_{steps}.pt")