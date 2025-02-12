# here we should run a simple app where you can enter a prompt and it returns a link to the wikipedia page
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path
import subprocess
import os
import argparse

from tinystories import TSDataset
from wikipedia import WikiDataset, WikiPrompt
from gpt import GPTModule
from transformers import GPT2Tokenizer


def main(parser):
    parser = GPTModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = GPTModule(config=args.config,
                      learning_rate=0,
                     betas=(0, 0),
                     weight_decay=0,
                     max_seq_len=args.max_seq_len,
                     tokenizer=tokenizer,
                     dataset_name=args.dataset,
                     overwrite_ds=True)

    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"])

    trainer = pl.Trainer.from_argparse_args(args, logger=TensorBoardLogger(os.path.join("data", "tb_logs"),
                                                                name=f"test_{args.config}_{args.dataset}"))

    if args.dataset == "ts":
        save_path = "data/tiny_stories_gpt4_encoded_%s_debug.lance"
        val_dataset = TSDataset(save_path, "validation", tokenizer, args.max_seq_len, tokenizer.eos_token_id)
        val_sampler = torch.utils.data.RandomSampler(val_dataset, True, val_dataset.__len__() // args.max_seq_len)

    elif args.dataset == "wiki":
        val_dataset = WikiDataset("validation", tokenizer, args.max_seq_len * args.seq_mult, tokenizer.eos_token_id)
        inds_val = np.arange(val_dataset.__len__(), step=1)
        val_sampler = torch.utils.data.SubsetRandomSampler(inds_val.astype(int).tolist())

    elif args.dataset == "wiki_prompt":
        val_dataset = WikiPrompt("validation", tokenizer, args.max_seq_len, args.seq_mult, tokenizer.eos_token_id)
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=0,
                                    sampler=val_sampler)

    model.ds = val_dataset.ds
    trainer.validate(model=model,
                     dataloaders=val_loader)
d


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--seq_mult', type=int, default=1)
    parser.add_argument('--dataset', type=str, help="ts|wiki", required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    main(parser)

