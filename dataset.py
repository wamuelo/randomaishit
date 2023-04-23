import torch
import os

class MyDataset(torch.utils.data.Dataset):
    CHARACTERS = []

    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        self.data = []

        # Read data from all files in directory
        for filename in os.listdir(data_path):
            with open(os.path.join(data_path, filename), 'r') as f:
                text = f.read()
            self.data += [text[i:i+self.seq_length] for i in range(0, len(text)-self.seq_length, self.seq_length)]

        # Map characters to indices
        self.chars = sorted(list(set(''.join(self.data))))
        self.char_to_idx = { ch:i for i,ch in enumerate(self.chars) }
        self.idx_to_char = { i:ch for i,ch in enumerate(self.chars) }
        
        # Set the class attribute CHARACTERS
        MyDataset.CHARACTERS = self.chars

    def __getitem__(self, index):
        x = [self.char_to_idx[ch] for ch in self.data[index]]
        y = [self.char_to_idx[ch] for ch in self.data[index][1:]]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.data)
