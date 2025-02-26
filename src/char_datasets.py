from abc import ABC, abstractmethod
from collections import defaultdict, Counter

import torch
from torch.utils import data

    
            

class CharWindowDataset(data.Dataset):
    def __init__(self, text: str, window_size: int = 1, max_map_size: int = 50, char_map = None, default_char = '~', device = "cpu") -> None:
        super().__init__()
        self.data = text
        self.device = device
        self.window_size = window_size
        if char_map is not None:
            self.map_size = len(char_map)
        else:
            self.map_size = min(max_map_size, len(set(text))+1)
        self.default_char = default_char
        self.set_maps(char_map)
        self.set_encoded_data()

    
    def set_maps(self, char_map):
        self.ctoi = defaultdict(lambda: self.map_size-1)
        if char_map is None:
            target_chars = Counter(self.data).most_common()[:self.map_size]
            char_map = [x[0] for x in target_chars]
            default_char_pos = char_map.index(self.default_char)
            if default_char_pos != -1:
                char_map[default_char_pos], char_map[-1] = char_map[-1], char_map[default_char_pos]
            else:
                char_map[-1] = self.default_char
        
        self.char_map = char_map
        self.ctoi.update({
            c:i for i, c in enumerate(char_map)
        })
        self.itoc = {
            i:c for i, c in enumerate(char_map)
        }
    
    def get_map(self):
        return self.char_map
        
    def __len__(self):
        return len(self.data) - self.window_size - 1
    
    def __getitem__(self, idx):
        X = self.encoded_data[idx:idx+self.window_size].to(self.device)
        y = self.encoded_data[idx+self.window_size].to(self.device)
        return X, y
    
    def encode(self, text):
        return torch.LongTensor([self.ctoi[c] for c in text]).to(self.device)
    
    def decode(self, text_tensor):
        return "".join(self.itoc[i] for i in text_tensor)
    
    def set_encoded_data(self):
        self.encoded_data = self.encode(self.data)
        


        