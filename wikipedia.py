import torch
from torch.utils.data import Dataset
import datasets
from random import randint
import lance

class WikiBase(Dataset):

    def get_toks(self, data, context_len):
        toks = self.tokenizer(data)
        ri = 0 #randint(0, max(len(toks.input_ids) - context_len, 0))
        ids = toks.input_ids[ri:ri+context_len]
        attn_mask = toks.attention_mask[ri:ri+context_len]
        pad_size = context_len - len(ids)
        ids = ids + [self.tokenizer.eos_token_id] * pad_size
        attn_mask = attn_mask + [0] * pad_size
        return ids, attn_mask


class WikiDataset(WikiBase): # TODO turn this into iterable?
    def __init__(self, split, tokenizer, context_len, eos_token_id):
        super().__init__()
        dataset = "data/datasets/wikipedia-train.arrow"

        self.ds = datasets.Dataset.from_file(dataset)

        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.context_len = context_len
        self.length = len(self.ds)
        self.name = "wiki"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.ds[idx]["text"] + self.tokenizer.eos_token
        ids, attn_mask = self.get_toks(data, self.context_len)
        # create metas
        metas = {**self.ds[idx], "query": "null"}
        return {
            "text_ids": torch.tensor(ids), 
            "text_attn": torch.tensor(attn_mask),
            "sample_id": int(self.ds[idx]["id"]),
            **metas
        }

class WikiPrompt(WikiBase): # TODO turn this into iterable?
    def __init__(self, split, tokenizer, context_len, seq_mul, eos_token_id):
        dataset = "data/datasets/prompt_wiki.lance"

        self.ds = lance.dataset(dataset)

        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.context_len = context_len
        self.seq_mul = seq_mul
        self.length = len(self.ds)
        self.name = "wiki_prompt"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        entry = self.ds.take([idx]).to_pandas().iloc[0].to_dict()
        data_text = entry["text"] + self.tokenizer.eos_token
        ids_text, attn_mask_text = self.get_toks(data_text, 
                                        self.context_len * self.seq_mul)
        data_prompt = entry["query"] + self.tokenizer.eos_token
        ids_prompt, attn_mask_prompt = self.get_toks(data_prompt, self.context_len)
        data_title = entry["title"] + self.tokenizer.eos_token
        ids_title, attn_mask_title = self.get_toks(data_title, self.context_len)
        return {
            "sample_id": int(entry["id"]),
            "text_ids": torch.tensor(ids_text),
            "text_attn": torch.tensor(attn_mask_text),
            "prompt_ids": torch.tensor(ids_prompt),
            "prompt_attn": torch.tensor(attn_mask_prompt),
            "title_ids": torch.tensor(ids_title),
            "title_attn": torch.tensor(attn_mask_title),
            **entry
        }

class WikiPromptFeats(WikiPrompt): # TODO turn this into iterable?
    def __init__(self, split, tokenizer, context_len, seq_mul, eos_token_id):
        super().__init__(split, tokenizer, context_len, seq_mul, eos_token_id)
        dataset = "data/datasets/wiki_prompt_val_feats.lance"
        self.ds = lance.dataset(dataset)

        # self.tokenizer = tokenizer
        # self.eos_token_id = eos_token_id
        # self.context_len = context_len
        # self.seq_mul = seq_mul
        # self.length = len(self.ds)
        self.name = "wiki_prompt_feats"

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        super_results = super().__getitem__(idx)
        entry = self.ds.take([idx]).to_pandas().iloc[0].to_dict()
        return {
            **super_results,
            "vector": entry['vector']
        }