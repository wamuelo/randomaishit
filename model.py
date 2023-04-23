import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h
