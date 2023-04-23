import torch.utils.data as data
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length
        with open(data_path, 'r') as f:
            text = f.read()
        chars = list(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        self.text = [self.char2idx[ch] for ch in text]

    def __getitem__(self, index):
        input_seq = self.text[index:index+self.seq_length]
        target_seq = self.text[index+1:index+self.seq_length+1]
        if len(input_seq) < self.seq_length:
            input_seq += [0] * (self.seq_length - len(input_seq))
            target_seq += [0] * (self.seq_length - len(target_seq))
        return np.array(input_seq), np.array(target_seq)

    def __len__(self):
        return len(self.text) - self.seq_length
