import os
import perf_event

import pandas as pd
import torch
from torch.cuda import device
from torch.utils.data import DataLoader
from torch_scatter import segment_max_coo as scatter_max
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding

from models.Coil import COIL
from models.Coil.coil_dataset import BenchmarkQueries
from utils.utils_general import create_this_perf

columns = ["encode_cycles", "encode_instructions",
           "encode_L1_misses", "encode_LLC_misses",
           "encode_L1_accesses", "encode_LLC_accesses",
           "encode_branch_misses", "encode_task_clock",
           "retrieval_cycles", "retrieval_instructions",
           "retrieval_L1_misses", "retrieval_LLC_misses",
           "retrieval_L1_accesses", "retrieval_LLC_accesses",
           "retrieval_branch_misses", "retrieval_task_clock"]

def dict_2_float(dd):
    for k in dd:
        dd[k] = dd[k].float()

    return dd

class CoilRetriever:
    def __init__(self, model_args, data_args, training_args, device):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.device = device
        self.query_dataset = None
        self.query_dataloader = None
        self.index = None
        self.model = None

    def set_model(self):
        config = AutoConfig.from_pretrained(
        self.model_args.model_name_or_path,
        num_labels=1
    )
        self.model = COIL.from_pretrained(
        self.model_args, self.data_args, self.training_args,
        self.model_args.model_name_or_path,
        from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
        config=config
    )

    def set_data_loader(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name,
            use_fast=False
        )
        self.query_dataset = BenchmarkQueries(self.data_args.dataset, self.tokenizer,
                                              self.data_args.query_json_dir,
                                              p_max_len=self.data_args.p_max_len)
        self.query_dataset.load_queries()

        self.query_dataloader = DataLoader(
            self.query_dataset,
            batch_size=1,
            collate_fn=DataCollatorWithPadding(
                self.tokenizer,
                max_length=self.data_args.p_max_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.training_args.dataloader_num_workers,
        )

    def load_index(self):
        index_load_path = r"{}/{}".format(self.data_args.doc_index_save_path, self.data_args.dataset)
        all_ivl_scatter_maps = torch.load(os.path.join(index_load_path, 'ivl_scatter_maps.pt'))
        all_shard_scatter_maps = torch.load(os.path.join(index_load_path, 'shard_scatter_maps.pt'))
        tok_id_2_reps = dict_2_float(torch.load(os.path.join(index_load_path, 'tok_reps.pt')))
        doc_cls_reps = torch.load(os.path.join(index_load_path, 'cls_reps.pt')).float()
        cls_ex_ids = torch.load(os.path.join(index_load_path, 'cls_ex_ids.pt'))
        self.index = (all_ivl_scatter_maps, all_shard_scatter_maps, tok_id_2_reps, doc_cls_reps, cls_ex_ids)

    def retrieve(self,topk=10):
        assert self.query_dataloader is not None
        assert self.index is not None
        assert self.model is not None
        assert self.query_dataset is not None
        perf_encode = perf_event.PerfEvent()
        perf_retrival = perf_event.PerfEvent()

        model = self.model.to(self.device)
        model.eval()
        all_ivl_scatter_maps, all_shard_scatter_maps, tok_id_2_reps, doc_cls_reps, cls_ex_ids = self.index
        all_query_match_scores = []
        all_query_inids = []
        all_perf = []
        for qid, batch in enumerate(tqdm(self.query_dataloader)):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    perf_encode.startCounters()
                    cls, reps = model.encode(**batch)
                    perf_encode.stopCounters()

            perf_retrival.startCounters()
            batch_qtok_ids = self.query_dataset.nlp_dataset[qid][:self.data_args.p_max_len - 2]
            batch_q_reps = cls.cpu()
            match_scores = torch.matmul(batch_q_reps, doc_cls_reps.transpose(0, 1))
            batched_tok_scores = []

            batch_qtok_ids = [q_id for q_id in batch_qtok_ids if q_id in all_ivl_scatter_maps.keys()]
            batch_qtok_ids = batch_qtok_ids + [self.tokenizer.sep_token_id]
            for batch_id, q_tok_id in enumerate(batch_qtok_ids):
                a = reps[0]
                q_tok_reps = reps[0][batch_id+1].unsqueeze(0)
                tok_reps = tok_id_2_reps[q_tok_id]
                tok_scores = torch.matmul(q_tok_reps, tok_reps.transpose(0, 1)).relu_()
                batched_tok_scores.append(tok_scores)

            for i, q_tok_id in enumerate(batch_qtok_ids):
                ivl_scatter_map = all_ivl_scatter_maps[q_tok_id]
                shard_scatter_map = all_shard_scatter_maps[q_tok_id]

                tok_scores = batched_tok_scores[i]
                ivl_maxed_scores = torch.empty(len(shard_scatter_map))

                for j in range(tok_scores.size(0)):
                    ivl_maxed_scores.zero_()
                    scatter_max(tok_scores[j], ivl_scatter_map, out=ivl_maxed_scores)
                    match_scores[0].scatter_add_(0, shard_scatter_map, ivl_maxed_scores)
            top_scores, top_iids = match_scores.topk(topk, dim=1)
            perf_retrival.stopCounters()
            all_query_match_scores.append(top_scores)
            all_query_inids.append(top_iids)
            this_perf = create_this_perf(perf_encode, perf_retrival)
            all_perf.append(this_perf)
        perf_df = pd.DataFrame(all_perf, columns=columns)
        scores = torch.cat(all_query_match_scores, dim=0)
        indices = torch.cat([cls_ex_ids[inids] for inids in all_query_inids], dim=0)
        return scores, indices, perf_df


