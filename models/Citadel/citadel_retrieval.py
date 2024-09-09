import collections
import json
import os
import perf_event

import ir_datasets
import ir_measures
import pandas as pd
import gzip
import torch
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer, DataCollatorWithPadding

from .citadel_dataloader import BenchmarkQueriesDataset, BenchmarkDataset
from dpr_scale.index.inverted_vector_index import IVFCPUIndex, IVFGPUIndex
from .citadel_model import CITADELEncoder
from .citadel_utils import process_check_point

class CitadelRetrieve:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.transformer_model_dir = self.config.transformer_model_dir
        self.checkpoint_path = self.config.check_point_path
        self.topk = config.topk
        self.perf_path = None
        self.rank_path = None
        self.eval_path = None
        self.context_encoder = None
        self.index = None
        self.device = device

    def setup(self):
        self._load_meta_data()
        checkpoint_dict = self._load_checkpoint()
        self._set_up_model(checkpoint_dict)
        self._sep_up_index()

    def _create_save_path(self):
        save_dir = r"{}/citadel/{}".format(self.config.results_save_to, self.config.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.perf_path = r"{}/{}".format(save_dir, "perf_results")
        self.rank_path = r"{}/{}".format(save_dir, "rank_results")
        self.eval_path = r"{}/{}".format(save_dir, "eval_results")
        if not os.path.exists(self.perf_path):
            os.makedirs(self.perf_path)
        if not os.path.exists(self.rank_path):
            os.makedirs(self.rank_path)
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path)

    def _load_meta_data(self):
        meta_data_path = r"{}/{}/metadata.json".format(self.config.index_dir, self.config.dataset)
        with open(meta_data_path, "r") as f:
            self.meta_data = json.load(f)

    def _load_checkpoint(self):
        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]
        checkpoint_dict = process_check_point(checkpoint_dict)
        return checkpoint_dict

    def _set_up_model(self, checkpoint_dict):
        self.context_encoder = CITADELEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)

        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _sep_up_index(self):

        if self.device == "cpu":
            self.index = IVFCPUIndex(self.config.portion, self.meta_data["corpus_len"],
                                     self.config.index_dir,
                                     self.config.dataset,
                                     self.config.prune_weight)
        elif self.device == "gpu":
            self.index = IVFGPUIndex(self.config.portion, self.meta_data["corpus_len"],
                                     self.config.index_dir,
                                     self.config.dataset,
                                     self.config.prune_weight,
                                     expert_parallel=False)

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

    def _retrieve(self):
        self._create_save_path()
        perf = perf_event.PerfEvent()
        all_query_match_scores = []
        all_query_inids = []
        all_perf = []
        for batch in tqdm(self.encode_loader):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        contexts_ids_dict[k] = v.to(self.config.encode_device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict
            queries_repr = self.context_encoder(batch, topk=1, add_cls=True)
            queries_repr = {k: v.detach().cpu() for k, v in queries_repr.items()}
            batch_embeddings = []
            batch_weights = []
            batch_cls = []
            if "cls_repr" in queries_repr:
                batch_cls = queries_repr["cls_repr"]
            embeddings = collections.defaultdict(list)
            weights = collections.defaultdict(list)
            for expert_repr, expert_topk_ids, expert_topk_weights, attention_score in zip(queries_repr["expert_repr"][0],
                                                                                        queries_repr["expert_ids"][0],
                                                                                        queries_repr["expert_weights"][0],
                                                                                        queries_repr["attention_mask"][0]):
                if attention_score > 0:
                    if len(queries_repr["expert_ids"].shape) == 2:
                        embeddings[expert_topk_ids.item()].append((expert_topk_weights * expert_repr).to(torch.float32))
                        weights[expert_topk_ids.item()].append(expert_topk_weights.to(torch.float32))
                    else:
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            if expert_weight > 0:
                                embeddings[expert_id.item()].append((expert_weight * expert_repr).to(torch.float32))
                                weights[expert_id.item()].append(expert_weight.to(torch.float32))
            batch_embeddings.append(embeddings)
            batch_weights.append(weights)
            perf.startCounters()
            batch_top_scores, batch_top_ids = self.index.search(batch_cls, batch_embeddings, batch_weights, self.topk)
            perf.stopCounters()
            all_query_match_scores.append(batch_top_scores)
            all_query_inids.append(batch_top_ids)
            cycles = perf.getCounter("cycles")
            instructions = perf.getCounter("instructions")
            L1_misses = perf.getCounter("L1-misses")
            LLC_misses = perf.getCounter("LLC-misses")
            L1_accesses = perf.getCounter("L1-accesses")
            LLC_accesses = perf.getCounter("LLC-accesses")
            branch_misses = perf.getCounter("branch-misses")
            task_clock = perf.getCounter("task-clock")
            all_perf.append([cycles, instructions,
                             L1_misses, LLC_misses,
                             L1_accesses, LLC_accesses,
                             branch_misses, task_clock])
            # return batch_top_scores.tolist(), batch_top_ids.tolist()
        # post processing
        all_query_match_scores = torch.cat(all_query_match_scores, dim=0)
        all_query_exids = torch.cat(all_query_inids, dim=0)
        self._save_perf(all_perf)
        path = self._save_ranks(all_query_match_scores, all_query_exids)
        return self.evaluate(path)

    def evaluate(self, path):
        qrels = ir_datasets.load(self.config.queries_path).qrels
        encode_dataset = BenchmarkDataset(self.config, None)
        new_2_old = list(encode_dataset.corpus.keys())
        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new_2_old[int(r["doc_id"])]
        eval_results = ir_measures.calc_aggregate(self.config.measure, qrels, rank_results_pd)
        eval_results["parameter"] = (str(self.config.prune_weight))
        eval_results["prune_weight"] = self.config.prune_weight
        return eval_results
        # eval_df = pd.DataFrame([eval_results])
        #
        # eval_df.to_csv(r"{}/eval-{}.csv".format(self.eval_path, str(self.config.prune_weight)), index=False)
        # print(eval_results)

    def _save_perf(self, all_perf: list):
        columns = ["cycles", "instructions",
                   "L1_misses", "LLC_misses",
                   "L1_accesses", "LLC_accesses",
                   "branch_misses", "task_clock"]
        perf_df = pd.DataFrame(all_perf, columns=columns)
        perf_df.to_csv(r"{}/prune_weight-{}.citadel_perf.csv".format(self.perf_path,
                                                                     str(self.config.prune_weight)),
                                                                        index=False)

    def _save_ranks(self, scores, indices):

        path = r"{}/prune_weight-{}.run.gz".format(self.rank_path, str(self.config.prune_weight))
        rh = faiss.ResultHeap(scores.shape[0], self.topk)
        if self.device == "cpu":
            rh.add_result(-scores.numpy(), indices.numpy())
        elif self.device == "gpu":
            scores = scores.to("cpu").to(torch.float32)
            indices = indices.to("cpu").to(torch.int64)
            rh.add_result(-scores.numpy(), indices.numpy())
        rh.finalize()
        corpus_scores, corpus_indices = (-rh.D).tolist(), rh.I.tolist()

        qid_list = list(self.dataset.queries.keys())
        with gzip.open(path, 'wt') as fout:
            for i in range(len(corpus_scores)):
                q_id = qid_list[i]
                scores = corpus_scores[i]
                indices = corpus_indices[i]
                for j in range(len(scores)):
                    fout.write(f'{q_id} 0 {indices[j]} {j} {scores[j]} run\n')
        return path

    def merge_eval(self):
        pass

    def run(self):
        self._prepare_data()
        eval_results = self._retrieve()
        return eval_results


# def merge_eval(eval_results_path):
#
#     for file in os.listdir(eval_results_path):
#         file_path = os.path.join(eval_results_path, file)
