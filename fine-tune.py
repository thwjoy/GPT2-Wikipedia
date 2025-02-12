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
from wikipedia import WikiPrompt, WikiPromptFeats
from gpt import GPTRecaller
from transformers import GPT2Tokenizer


def main(parser):
    parser = GPTRecaller.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # TODO save location?

    model = GPTRecaller(
        distance_metric=args.distance_metric,
        query_ckpt=args.query_ckpt,
        query_config=args.query_config,
        feat_ckpt=args.feat_ckpt,
        feat_config=args.feat_config,
        tokenizer=tokenizer,
        learning_rate=args.learning_rate,
        betas=args.betas,
        weight_decay=args.weight_decay
    )

    run_name = Path(args.query_config).stem + '_' + args.dataset + '_' + subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                        monitor="val_loss",
                                        save_top_k=3,
                                        dirpath=os.path.join(args.default_root_dir, run_name),
                                        filename='{epoch:02d}'
                                    )
                                   
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=TensorBoardLogger("data/tb_logs", name=f"fine_tune_{args.query_config}_{args.dataset}"))

    dataset = WikiPromptFeats("train", tokenizer, args.max_seq_len, 1, tokenizer.eos_token_id)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=0)

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)



if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--seq_mult', type=int, default=1)
    parser.add_argument('--dataset', type=str, help="ts|wiki", required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-5)
    parser.add_argument('--betas', nargs='+', default=(0.9, 0.95))
    parser.add_argument('--weight_decay', type=float, default=0.1)
    main(parser)

