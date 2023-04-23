import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, proj_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, proj_size=proj_size)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.shape[0])

        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)

        return out, h

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers*2, batch_size, self.hidden_size),
                torch.zeros(self.num_layers*2, batch_size, self.hidden_size))