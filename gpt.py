import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import optim, nn, utils
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F

import transformers
from transformers import GPT2Config, \
                         GPT2LMHeadModel, \
                         GPT2Tokenizer, \
                         AutoModelForCausalLM, \
                         GPTNeoConfig

from tqdm.auto import tqdm
import lance
from lance.vector import vec_to_table
import pyarrow as pa
import json 
import os
import shutil
import numpy as np

class GPTModule(pl.LightningModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("GPTModel")
        parser.add_argument("--config", type=str, help="path to config", required=True)
        return parent_parser

    def __init__(self, learning_rate, betas, weight_decay, config, max_seq_len, tokenizer, dataset_name, overwrite_ds=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.config = config
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        with open(self.config, "r") as f:
            conf = GPTNeoConfig(**json.load(f))
        conf.pad_token_id = self.tokenizer.pad_token_id
        self.gpt = AutoModelForCausalLM.from_config(conf)
        self.val_epochs = 0
        self.k = 5
        self.db_uri = f"data/datasets/{self.dataset_name}_val_feats.lance"
        assert self.dataset_name in ['ts', 'wiki', 'wiki_prompt'], f"Dataset \"{self.dataset_name}\" not recognised"
        if dataset_name == "ts":
            self.query = "Lily and Ben are friends. They like to play in the park."
        elif dataset_name == "wiki" or dataset_name == "wiki_prompt":
            self.query = "Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy.."
        else:
            self.query = ""
        self.overwrite_ds = overwrite_ds
        # TODO save hyper params/sort all this out it's gross
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx): 
        x, attn_mask = batch["text_ids"], batch["text_attn"]
        loss = self.gpt(input_ids=x,
                        attention_mask=attn_mask,
                        labels=x).loss
        self.log("train_loss", loss)
        return loss
        
    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        if self.overwrite_ds:
            shutil.rmtree(self.db_uri, ignore_errors=True)
            df = pd.DataFrame(columns=['id', 'vector'])
            self.schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("url", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("query", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.gpt.config.hidden_size)),
                ])
            
            tbl = pa.Table.from_pydict({"id": [],
                                        "url": [],
                                        "title": [],
                                        "text": [],
                                        "query": [],
                                        "vector": []}, schema=self.schema)
            lance.write_dataset(tbl, self.db_uri)

    def validation_step(self, batch, batch_idx):
        super().validation_step()
        x, attn_mask = batch["text_ids"], batch["text_attn"]
        n_chunks = x.shape[1] // self.max_seq_len
        hidden_states = torch.zeros(attn_mask.shape[0],
                                    self.gpt.config.hidden_size,
                                    device=x.device)
        loss = 0
        for c in range(n_chunks): # TODO make this nicer
            data = x[:, c:c+self.max_seq_len]
            attn = attn_mask[:, c:c+self.max_seq_len]
            out = self.gpt(input_ids=data,
                            attention_mask=attn,
                            labels=data,
                            output_hidden_states=True)
            hidden_states += torch.sum(out.hidden_states[-1].clone() * attn.unsqueeze(-1), dim=1)
            loss += out.loss
            del out
        self.log("val_loss", loss / n_chunks)
        if self.overwrite_ds:
            mean_hidden = hidden_states / attn_mask.sum(dim=1).unsqueeze(1)
            metas = {}
            metas['vector'] = [np.array(h) for h in mean_hidden.cpu().numpy().squeeze().tolist()]
            inds = [batch['id'].index(str(i.item())) for i in batch["sample_id"].cpu()]
            metas['id'] = [int(idx) for idx in np.array(batch['id'])[inds]]
            for key in ['url', 'title', 'text', 'query']:
                metas[key] = [batch[key][i] for i in inds]
            lance.write_dataset(pa.Table.from_pydict(metas, schema=self.schema),
                                self.db_uri, mode="append")
       
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        inputs = self.tokenizer(self.query, return_tensors="pt")
        generations = self.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        self.logger.experiment.add_text("Generations", generations, global_step=self.val_epochs)

        if os.path.isdir(self.db_uri):
            results = self.find_top_k(self.query, self.k)
            if results:
                for i in range(len(results['vector'])):
                    to_print = f"{results['title'][i]}\r\n{results['url'][i]}\r\n\r\n{results['text'][i]}"
                    self.logger.experiment.add_text(f"Lookups {i}", to_print, global_step=self.val_epochs)
        self.val_epochs += 1
                
    def find_top_k(self, query, k):
        if os.path.isdir(self.db_uri):
            db = lance.dataset(self.db_uri)
            query_toks = self.tokenizer(query, return_tensors="pt")
            query_feats = self.gpt(input_ids=query_toks["input_ids"].to(self.device),
                                attention_mask=query_toks["attention_mask"].to(self.device),
                                output_hidden_states=True).hidden_states[-1].mean(dim=1).cpu()
            ret = db.to_table(nearest={"column": "vector", "k": k, "q": query_feats.view(-1).tolist()}).to_pandas().to_dict()
            return ret
        else:
            return None

    def generate(self, input_ids, attention_mask):
        generation_output = self.gpt.generate(input_ids=input_ids.to(self.device),
                                              attention_mask=attention_mask.to(self.device),
                                              max_new_tokens=self.max_seq_len - input_ids.shape[0],
                                              pad_token_id=self.tokenizer.eos_token_id,
                                              return_dict_in_generate=True,
                                              output_scores=True).sequences
        return self.tokenizer.decode(generation_output[0])

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,  transformers.Conv1D)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.gpt.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.gpt.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas)
        return optimizer


class GPTRecaller(pl.LightningModule):

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("GPTModel")
        parser.add_argument("--distance_metric", type=str, help="cosine|L2", default="cosine")
        parser.add_argument("--query_ckpt", type=str, help="for query module", required=True)
        parser.add_argument("--query_config", type=str, required=True)
        parser.add_argument("--feat_ckpt", type=str, help="for feature module", required=True)
        parser.add_argument("--feat_config", type=str, required=True)
        parser.add_argument("--spare_train", action="store_true")
        return parent_parser

    @staticmethod
    def parse_state_dict(d):
        empty = {}
        for k, v in d.items():
            empty[k[4:]] = v
        return empty

    def __init__(self, distance_metric, query_ckpt, query_config, feat_ckpt, feat_config, tokenizer, 
                 learning_rate, betas, weight_decay):
        super().__init__()
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.db_uri = "data/datasets/wiki_prompt_val_feats.lance"
        self.query = "Football"
        self.k = 5
        self.val_epochs = 0

        with open(query_config, "r") as f:
            conf = GPTNeoConfig(**json.load(f))
        conf.pad_token_id = self.tokenizer.pad_token_id
        self.query_model = AutoModelForCausalLM.from_config(conf)

        self.query_model.load_state_dict(GPTRecaller.parse_state_dict(
            torch.load(query_ckpt, map_location="cpu")["state_dict"]))

        with open(feat_config, "r") as f:
            conf = GPTNeoConfig(**json.load(f))
        conf.pad_token_id = self.tokenizer.pad_token_id
        self.feat_model = AutoModelForCausalLM.from_config(conf)
        self.feat_model.load_state_dict(GPTRecaller.parse_state_dict(
            torch.load(query_ckpt, map_location="cpu")["state_dict"]))

    def configure_optimizers(self):
        params = list(self.query_model.parameters())
        return torch.optim.AdamW(params, lr=self.learning_rate, betas=self.betas)

    def training_step(self, batch):
        # TODO this will be wrong
        text_ids, text_attn = batch['text_ids'], batch['text_attn']
        prompt_ids, prompt_attn = batch['prompt_ids'], batch['prompt_attn']

        query_outputs = self.query_model(input_ids=prompt_ids,
                                         attention_mask=prompt_attn,
                                         output_hidden_states=True)
        query_feats = torch.sum(query_outputs.hidden_states[-1] * prompt_attn.unsqueeze(-1), dim=1) / prompt_attn.sum(dim=-1).unsqueeze(1)


        # TODO  drop this, we should just recive the features
        if "vector" in batch.keys():
            text_feats = batch["vector"]
        else:
            with torch.no_grad():
                text_outputs = self.feat_model(input_ids=text_ids,
                                                attention_mask=text_attn,
                                                output_hidden_states=True)
                text_feats = torch.sum(text_outputs.hidden_states[-1] * text_attn.unsqueeze(-1), dim=1) / text_attn.sum(dim=-1).unsqueeze(1)
     

        # import pdb; pdb.set_trace()

        # do contrastive loss on cosine distance, minimise and maximise cosine distance
        bs = text_feats.shape[0]
        targs = torch.eye(bs, device=text_feats.device)
        pos_weight = (bs**2 - bs) / bs
        weight = -1 * torch.ones_like(targs)
        weight.fill_diagonal_(pos_weight)
        contrastives = torch.matmul(text_feats / text_feats.norm(dim=-1, keepdim=True),
                                    (query_feats / query_feats.norm(dim=-1, keepdim=True)).t())

        loss = -1 * contrastives.view(-1).mean(dim=0)
        self.log("train_los", loss)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # load the dataset, need to do some logic to load the right dataset
        # self.ds = lance.dataset(f"data/datasets/{self.dataset_name}_prompt_val_feats.lance")


    def validation_step(self, batch, *args, **kwargs):
        if os.path.isdir(self.db_uri):
            results = self.find_top_k(self.query, self.k)
            if results:
                for i in range(len(results['vector'])):
                    to_print = f"{results['title'][i]}\r\n{results['url'][i]}\r\n\r\n{results['text'][i]}"
                    self.logger.experiment.add_text(f"Lookups {i}", to_print, global_step=self.val_epochs)
        self.val_epochs += 1

    def find_top_k(self, query, k):
        if os.path.isdir(self.db_uri):
            db = lance.dataset(self.db_uri)
            query_toks = self.tokenizer(query, return_tensors="pt")
            query_feats = self.query_model(input_ids=query_toks["input_ids"].to(self.device),
                                attention_mask=query_toks["attention_mask"].to(self.device),
                                output_hidden_states=True).hidden_states[-1].mean(dim=1).cpu()
            ret = db.to_table(nearest={"column": "vector",
                                       "k": k,
                                       "q": query_feats.view(-1).tolist(),
                                       "metric": "cosine"}).to_pandas().to_dict()
            return ret
        else:
            return None

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        
