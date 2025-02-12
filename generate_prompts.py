# %%
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch 

tokenizer = LlamaTokenizer.from_pretrained("llama_7b", padding_side='left')
model = LlamaForCausalLM.from_pretrained("llama_7b").to("cuda:0")
max_seq_len = 256
batch_size = 1

# %%
from wikipedia import WikiDataset
import numpy as np
import torch

train_dataset = WikiDataset(None, "train", tokenizer, 2048 - max_seq_len, tokenizer.eos_token_id)
inds_train = np.arange(train_dataset.__len__(), step=1500)
train_sampler = torch.utils.data.SubsetRandomSampler(inds_train.astype(int).tolist())
train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    sampler=train_sampler)

# %%
pretext = "You are a helpful assistant. Summarize:"
pre_toks = tokenizer(pretext, return_tensors="pt")["input_ids"].expand(batch_size, -1)

# %%
# set up dataset schema
import datasets
import pyarrow as pa
import lance
import shutil

db_uri = "prompt_wiki.lance"
shutil.rmtree(db_uri, ignore_errors=True)
ds = datasets.Dataset.from_file("data/wikipedia-train.arrow")
schema = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("url", pa.string()),
        pa.field("title", pa.string()),
        pa.field("text", pa.string()),
        pa.field("query", pa.string()),
    ])
tbl = pa.Table.from_pydict({"id": [],
                            "url": [],
                            "title": [],
                            "text": [],
                            "query": []},
                            schema=schema)
lance.write_dataset(tbl, db_uri)


# %%
from tqdm import tqdm
import pandas as pd

for toks, mask, metas in tqdm(iter(train_loader)): # TODO seqneuce length and add summary 
    toks = torch.cat((pre_toks, toks[:, 1:-1]), dim=-1)
    mask = torch.cat((torch.ones_like(pre_toks), mask[:, 1:-1]), dim=-1)
    assert (toks.shape == mask.shape), f"Shapes {mask.shape} {toks.shape}" 
    generation_output = model.generate(input_ids=toks.to("cuda:0"),
                                    attention_mask=mask.to("cuda:0"),
                                    max_new_tokens=max_seq_len,
                                    pad_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=True,
                                    output_scores=False,
                                    do_sample=True,
                                    temperature=1.2,
                                    ).sequences

    # here we need to save the summary
    query = tokenizer.decode(generation_output[0][toks.shape[-1]:])
    df = pd.DataFrame({**metas, "query": query})
    lance.write_dataset(pa.Table.from_pandas(df), db_uri, mode="append")



