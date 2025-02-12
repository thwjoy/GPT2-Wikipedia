
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from pathlib import Path
import subprocess
import os

from tinystories import TSDataset
from wikipedia import WikiDataset
from gpt import GPTModule
from transformers import GPT2Tokenizer


def main(parser):
    parser = GPTModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # TODO can probably trim this

    model = GPTModule(config=args.config,
                      learning_rate=args.learning_rate,
                      betas=args.betas,
                      weight_decay=args.weight_decay,
                      max_seq_len=args.max_seq_len,
                      tokenizer=tokenizer,
                      dataset_name=args.dataset,
                      overwrite_ds=False)



    run_name = Path(args.config).stem + '_' + args.dataset + '_' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                        monitor="val_loss",
                                        save_top_k=3,
                                        dirpath=os.path.join(args.default_root_dir, run_name),
                                        filename='{epoch:02d}'
                                    )

    trainer = pl.Trainer.from_argparse_args(args,
                                            callbacks=[checkpoint_callback],
                                            logger=TensorBoardLogger(os.path.join("data", "tb_logs"),
                                                                     name=run_name))

    if args.dataset == "ts":
        save_path = "data/tiny_stories_gpt4_encoded_%s.lance"
        train_dataset = TSDataset(save_path, "train", tokenizer, args.max_seq_len, tokenizer.eos_token_id)
        val_dataset = TSDataset(save_path, "validation", tokenizer, args.max_seq_len, tokenizer.eos_token_id)
        train_sampler = torch.utils.data.RandomSampler(train_dataset, True, train_dataset.__len__() // args.max_seq_len)
        val_sampler = torch.utils.data.RandomSampler(val_dataset, True, val_dataset.__len__() // args.max_seq_len)

    elif args.dataset == "wiki":
        train_dataset = WikiDataset("train", tokenizer, args.max_seq_len, tokenizer.eos_token_id)
        val_dataset = WikiDataset("validation", tokenizer, args.max_seq_len, tokenizer.eos_token_id)
        inds_train = np.arange(train_dataset.__len__(), step=100)
        train_sampler = torch.utils.data.SubsetRandomSampler(inds_train.astype(int).tolist())
        inds_val = np.arange(train_dataset.__len__(), step=100)
        val_sampler = torch.utils.data.SubsetRandomSampler(inds_val.astype(int).tolist())


    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=0,
                                        sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=0,
                                    sampler=val_sampler)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

if __name__ == "__main__":
    import argparse
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--betas', nargs='+', default=(0.9, 0.95))
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, help="ts|wiki")
    main(parser)
