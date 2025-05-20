# dataset
import torch
from torch.utils.data import Dataset

class FPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        encoded = {key: val.squeeze(0) for key, val in encoded.items()}
        encoded["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return encoded