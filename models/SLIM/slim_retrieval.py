import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding

from dataset.BenchmarkDataset import BenchmarkQueriesDataset
from models.SLIM.slim_encoder import SlimEncoder


class SLIMRetrieval:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.transformer_model_dir = self.config.transformer_model_dir
        self.checkpoint_path = self.config.check_point_path
        self.topk = config.topk

    def _load_checkpoint(self):
        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]
        return checkpoint_dict

    def _set_up_model(self, checkpoint_dict):
        self.context_encoder = SlimEncoder(model_path=self.config.transformer_model_dir,
                                           dropout=self.config.dropout)
        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _prepare_data(self):
        tokenizer = BertTokenizer.from_pretrained(self.transformer_model_dir, use_fast=False)
        self.dataset = BenchmarkQueriesDataset(self.config, tokenizer)
        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=self.config.max_seq_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.config.dataloader_num_workers,
        )


    def setup(self):
        checkpoint_dict = self._load_checkpoint()
        self._set_up_model(checkpoint_dict)
        self._sep_up_index()

