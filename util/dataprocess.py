"""
Copyright (c) 2024 albert-jeffery. All rights reserved.
Project: DeepSpeed-Finetuning
GitHub: https://github.com/albert-jeffery/DeepSpeed-Finetuning
"""

from dataclasses import dataclass
from typing import List, Union, Optional, Any, Dict, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
from tqdm import tqdm

@dataclass
class CustomDataCollator:
    r"""
    Data collator for custom dataset.
    Args:
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
    """
    tokenizer: Optional[AutoTokenizer] = None

    def __call__(self, batch: Optional[Tuple]) -> Union[Dict, List[Tuple]]:
        src_batch, trg_batch = [], []
        for (src, trg) in batch:
            src = torch.tensor(src)
            trg = torch.tensor(trg)
            src_batch.append(src)
            trg_batch.append(trg)
        src_padding = self.tokenizer.pad_token_id

        return pad_sequence(src_batch, batch_first=True, padding_value=src_padding), \
               pad_sequence(trg_batch, batch_first=True, padding_value=-100)
    
class CustomDataset(Dataset):
    r"""
    Dataset class for custom dataset.
    Args:
        data_path (str): Path to the data file.
        tokenizer (AutoTokenizer): Tokenizer for the dataset.
        src_len (int, optional): Maximum length of the source sequence. Defaults to 512.
        trg_len (int, optional): Maximum length of the target sequence. Defaults to 128.
    """
    def __init__(
            self, 
            data_path: str, 
            tokenizer: Optional[AutoTokenizer], 
            src_len: int = 512, 
            trg_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.src_len = src_len
        self.trg_len = trg_len
        self.data_temp = []
        self.max_len = src_len + trg_len

        with open(data_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f.readlines(), desc='loading data...'):
                data = json.loads(line)
                src_ids = self.tokenizer.encode(data['text'], add_special_tokens=False)
                trg_ids = self.tokenizer.encode(data['summary'], add_special_tokens=False)
                if len(src_ids) > (self.src_len - 1):
                    src_ids = src_ids[:self.src_len - 1]
                if len(trg_ids) > (self.trg_len - 2):
                    trg_ids = trg_ids[:self.trg_len - 2]

                # input_ids = src_ids + [gMASK] + <sop> + trg_ids
                tokenizer.encode([1,2,3], add_special_tokens=False)
                input_ids = src_ids + [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")] + trg_ids
                
                # labels = [-100] * len(src_ids + 1) + trg_ids + <eos>
                mask_position = input_ids.index(tokenizer.get_command("sop")) 
                labels = [-100] * mask_position + input_ids[mask_position + 1:] + [tokenizer.eos_token_id]
                
                self.data_temp.append((input_ids, labels))

    def __getitem__(self, index):
        return self.data_temp[index]
    
    def __len__(self): 
        return len(self.data_temp)
