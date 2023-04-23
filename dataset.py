import os

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        self.chars = []
        self.data = []

        # Read data from all files in directory
        for filename in os.listdir(data_path):
            with open(os.path.join(data_path, filename), 'r') as f:
                text = f.read()
            self.chars += list(set(text))
            self.data += [text[i:i+self.seq_length] for i in range(0, len(text)-self.seq_length, self.seq_length)]

        # Map characters to indices
        self.char_to_idx = { ch:i for i,ch in enumerate(sorted(self.chars)) }
        self.idx_to_char = { i:ch for i,ch in enumerate(sorted(self.chars)) }

    def __getitem__(self, index):
        x = [self.char_to_idx[ch] for ch in self.data[index]]
        y = [self.char_to_idx[ch] for ch in self.data[index][1:]]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.data)