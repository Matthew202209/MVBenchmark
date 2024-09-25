import argparse
import collections
import gzip
import json
import os

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.hipify.hipify_python import meta_data
from tqdm import tqdm

import ir_measures
from ir_measures import *

from models.Citadel.citadel_dataloader import CitadelQueryDataset, CitadelDataset
from models.Citadel.citadel_model import CITADELEncoder
from models.Citadel.citadel_searcher import IVFCPUIndex
from models.Citadel.citadel_transformer import HFTransform
from models.Citadel.citadel_utils import process_check_point


class CitadelRetrieve:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.transformer_model_dir = self.config.transformer_model_dir
        self.checkpoint_path = self.config.check_point_path
        self.topk = config.content_topk
        self.perf_path = None
        self.rank_path = None
        self.eval_path = None
        self.context_encoder = None
        self.index = None
        self.metadata = None
        self.device = device


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
        self.index = IVFCPUIndex(self.config.portion, self.meta_data["num_docs"], self.config.index_dir,
                                 self.config.dataset, self.config.content_topk, self.config.prune_weight)

    def setup(self):
        self._load_meta_data()
        checkpoint_dict = self._load_checkpoint()
        self._set_up_model(checkpoint_dict)
        self._sep_up_index()

    def _load_meta_data(self):
        meta_data_path = r"{}/{}.json".format(self.config.ctx_embeddings_dir, self.config.dataset)
        with open(meta_data_path, "r") as f:
            self.meta_data = json.load(f)

    def _prepare_data(self):
        transform = HFTransform(self.config.transformer_model_dir, self.config.max_seq_len)
        self.dataset = CitadelQueryDataset(self.config, transform)

        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )


    def _retrieve(self):
        self._create_save_path()
        all_query_match_scores = []
        all_query_inids = []
        for batch in tqdm(self.encode_loader):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        contexts_ids_dict[k] = v.to(self.config.encode_device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict

            queries_repr = self.context_encoder(batch, topk=1, add_cls=True)
            queries_repr = {k: v.detach() for k, v in queries_repr.items()}
            batch_embeddings = []
            batch_weights = []
            batch_cls = []

            if "cls_repr" in queries_repr:
                batch_cls = queries_repr["cls_repr"]

            for batch_id in range(1):
                embeddings = collections.defaultdict(list)
                weights = collections.defaultdict(list)
                for expert_repr, expert_topk_ids, expert_topk_weights, attention_score in zip(
                        queries_repr["expert_repr"][batch_id],
                        queries_repr["expert_ids"][batch_id],
                        queries_repr["expert_weights"][batch_id],
                        queries_repr["attention_mask"][batch_id]):
                    if attention_score > 0:
                        if len(queries_repr["expert_ids"].shape) == 2:
                            embeddings[expert_topk_ids.item()].append(
                                (expert_topk_weights * expert_repr).to(torch.float32))
                            weights[expert_topk_ids.item()].append(expert_topk_weights.to(torch.float32))
                        else:
                            for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                                if expert_weight > 0:
                                    embeddings[expert_id.item()].append((expert_weight * expert_repr).to(torch.float16))
                                    weights[expert_id.item()].append(expert_weight.to(torch.float16))
                batch_embeddings.append(embeddings)
                batch_weights.append(weights)

            batch_top_scores, batch_top_ids = self.index.search(batch_cls, batch_embeddings, batch_weights, self.topk)
            all_query_match_scores.append(batch_top_scores)
            all_query_inids.append(batch_top_ids)



        all_query_match_scores = np.concatenate(all_query_match_scores, axis=0)
        all_query_exids = np.concatenate(all_query_inids, axis=0)

        path = self._save_ranks(all_query_match_scores, all_query_exids)
        return path

    def run(self):
        self.setup()
        self._prepare_data()
        path = self._retrieve()
        self.evaluate(path)

    def evaluate(self, path, index_memory=None):
        qrels = pd.read_csv(r"{}/{}.csv".format(self.config.label_json_dir, self.config.dataset))
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["doc_id"] = qrels["doc_id"].astype(str)

        encode_dataset = CitadelDataset(self.config, None)
        new_2_old = list(encode_dataset.corpus.keys())
        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new_2_old[int(r["doc_id"])]
        eval_results = ir_measures.calc_aggregate(self.config.measure, qrels, rank_results_pd)
        eval_results["parameter"] = (str(self.config.prune_weight))
        eval_results["prune_weight"] = self.config.prune_weight
        eval_results["index_memory"] = index_memory
        eval_results["index_time"] = self.meta_data["index_time"]
        eval_results["num_docs"] = self.meta_data["num_docs"]
        eval_results["avgg_num_tokens"] = self.meta_data["avgg_num_tokens"]
        eval_results["total_num_tokens"] = self.meta_data["total_num_tokens"]
        print(eval_results)
        return eval_results

    def _save_ranks(self, scores, indices):
        path = r"{}/Citadel.run.gz".format(self.rank_path)
        rh = faiss.ResultHeap(scores.shape[0], self.topk)
        if self.device == "cpu":
            # rh.add_result(-scores.numpy(), indices.numpy())
            rh.add_result(-scores, indices)
        else:
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