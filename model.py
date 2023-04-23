import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(MyModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden=None):
        embedded = self.embed(inputs)
        outputs, hidden = self.lstm(embedded, hidden)
        outputs = self.linear(outputs.view(-1, self.linear.in_features))
        return outputs, hidden
