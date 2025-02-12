import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm
import lance
import numpy as np
import json 
import os


def tokenize_data_to_lance(tokenizer, d_path, save_path, inc_prompts=False):
    def make_lance(chunk_paths, split):
        all_tokens = []
        for c, chunk_path in enumerate(chunk_paths):
            with open(chunk_path, 'r') as f:
                chunk = json.load(f)
            print(f"Chunk {c}")
            for story in tqdm(chunk):
                if inc_prompts:
                    doc = story["instruction"]["prompt:"] + tokenizer.eos_token
                    encoded_prompt = tokenizer(doc)['input_ids']
                    all_tokens.extend(encoded_prompt)
                doc = story["story"] + tokenizer.eos_token
                doc = doc.replace("\n", "")
                encoder_story = tokenizer(doc)['input_ids']
                all_tokens.extend(encoder_story)

        pa_table = pa.Table.from_arrays([all_tokens], names=['value'])
        lance.write_dataset(pa_table, save_path % split, {'model': 'create'}, mode='overwrite')
        print(f"Total tokens in {split} tokenized dataset: {len(all_tokens):,.0f}")
    
    tokenize_chunk_paths = [os.path.join(d_path, fn) for fn in os.listdir(d_path) if fn.endswith('.json')]
    split_frac = int(0.9 * len(tokenize_chunk_paths))
    make_lance(tokenize_chunk_paths[:split_frac], "train")
    make_lance(tokenize_chunk_paths[split_frac:], "validation")
    
class TSDataset(Dataset): # TODO turn this into iterable?
    def __init__(self, save_path, split, tokenizer, context_len, eos_token_id):
        dataset = "data/TinyStories_all_data"
        if not os.path.exists(save_path % split):
            tokenize_data_to_lance(tokenizer, dataset, save_path, False)

        self.eos_token_id = eos_token_id
        self.ds = lance.dataset(save_path % split)
        self.context_len = context_len
        self.length = self.ds.count_rows() - context_len
        self.name = "ts"

    def __len__(self):
        return self.length
    
    def from_idxs(self, idxs):
        data = self.ds.take(idxs).to_pylist()
        data = torch.tensor(list(map(lambda x: x['value'], data)))
        return data

    def __getitem__(self, idx):
        current_window_idxs = np.arange(idx, idx+self.context_len)
        data = self.from_idxs(current_window_idxs)
        attn_mask = (torch.isin(data, self.eos_token_id).cumsum(0) == 0).float()
        pad_inds = (attn_mask == 0).nonzero().squeeze()
        data[pad_inds] = self.eos_token_id
        attn_mask = attn_mask.roll(1) # shift values to eos is included in attention
        attn_mask[0] = 1
        return data, attn_mask, {}