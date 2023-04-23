import torch
import json
import numpy as np
from model import MyModel
import argprase

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='Name of the saved model file')

args = parser.parse_args()

with open('config.json', 'r') as f:
    config = json.load(f)

model_path = config['model_path']
seq_length = config['seq_length']

# Load dataset
data_path = config['data_path']
with open(f'{data_path}/char2idx.json', 'r') as f:
    char2idx = json.load(f)
idx2char = {i: ch for ch, i in char2idx.items()}
vocab_size = len(char2idx)

# Load model
model = MyModel(vocab_size, config['hidden_size'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(f'{model_path}/{args.model_name}.pt', map_location=device))
model.eval()

# Generate text
start_token = 'I am'
temperature = 0.5
with torch.no_grad():
    input_seq = torch.tensor([char2idx[ch] for ch in start_token], dtype=torch.long).unsqueeze(1).to(device)
    output_seq = [ch for ch in start_token]

    hidden = None
    for i in range(seq_length - len(start_token)):
        output, hidden = model(input_seq, hidden)
        probs = torch.softmax(output / temperature, dim=0).cpu().numpy()
        next_char_idx = np.random.choice(len(idx2char), p=probs)
        next_char = idx2char[next_char_idx]
        output_seq.append(next_char)

        input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    output_text = ''.join(output_seq)
    print(output_text)
