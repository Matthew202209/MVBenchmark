import argparse
import collections
import concurrent.futures
import glob
import json
import os
import pickle
import shutil
from copy import deepcopy
from os import truncate
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.Citadel.citadel_dataloader import CitadelDataset
from models.Citadel.citadel_model import CITADELEncoder
from models.Citadel.citadel_transformer import HFTransform
from models.Citadel.citadel_utils import process_check_point


class CitadelIndex:
    def __init__(self, config):
        self.config = config
        self.transformer_model_dir = config.transformer_model_dir
        self.ctx_embeddings_dir = config.ctx_embeddings_dir
        self.latency = collections.defaultdict(float)
        self.content_topk = str(config.content_topk)
        self.meta_data = None
        self.encode_loader = None
        self.context_encoder = None
        self.dataset = None

    def setup(self):
        self._prepare_data()
        self._prepare_model()

    def run(self):
        batch_results_list, batch_cls_list = self._encode()
        self.save_encode(batch_results_list, batch_cls_list)
        self.save_metadata()

    def save_metadata(self):
        metadata_file_path = r"{}/{}/metadata.json".format(self.ctx_embeddings_dir, self.config.dataset)
        self._load_meta_data()
        self.meta_data["index_time"] = self.latency["index_time"]

        with open(metadata_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.meta_data, json_file, ensure_ascii=False, indent=4)

    def _prepare_data(self):
        transform = HFTransform(self.config.transformer_model_dir, self.config.max_seq_len)
        self.dataset = CitadelDataset(self.config, transform)
        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=self.config.encode_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )

    def _prepare_model(self):
        self.context_encoder = CITADELEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)

        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]

        checkpoint_dict = process_check_point(checkpoint_dict)
        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _encode(self):
        batch_results_list = []
        batch_cls_list = []
        start_time = time.time()
        for batch in tqdm(self.encode_loader):
            corpus_ids = list(batch.data["corpus_ids"][0])
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        if k == "corpus_ids":
                            continue
                        contexts_ids_dict[k] = v.to(self.config.device)
                    batch.data = contexts_ids_dict
                    del contexts_ids_dict
                    contexts_repr = self.context_encoder(batch, topk=int(self.content_topk), add_cls=True)
            contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items()}
            batch_results = []
            batch_cls = []
            if "cls_repr" in contexts_repr:
                batch_cls = contexts_repr["cls_repr"]
            for batch_id, corpus_id in enumerate(corpus_ids):
                results = collections.defaultdict(list)
                for expert_repr, expert_topk_ids, expert_topk_weights, attention_score, context_id in zip(
                        contexts_repr["expert_repr"][batch_id],
                        contexts_repr["expert_ids"][batch_id],
                        contexts_repr["expert_weights"][batch_id],
                        contexts_repr["attention_mask"][batch_id],
                        batch.data["input_ids"][batch_id][1:]):
                    if attention_score > 0:
                        if len(contexts_repr["expert_ids"].shape) == 2:  # COIL and ColBERT
                            if expert_topk_weights > 0:
                                results[expert_topk_ids.item()].append(
                                    [int(corpus_id), expert_topk_weights, expert_topk_weights * expert_repr])
                        else:  # CITADEL
                            for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                                if expert_weight > self.config.weight_threshold:
                                    results[expert_id.item()].append(
                                        [int(corpus_id), expert_weight, expert_weight * expert_repr])
                batch_results.append(results)
            batch_results_list.append(batch_results)
            batch_cls_list.append(batch_cls)
        end_time = time.time()
        self.latency["index_time"] = end_time - start_time
        return batch_results_list, batch_cls_list

    def load_context_expert(self, expert_file_name):
        def load_file(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data

        data = []

        input_paths = sorted(glob.glob(os.path.join(self.config.ctx_embeddings_dir,
                                                    self.config.dataset,
                                                    self.content_topk,
                                                    r"expert_original",
                                                    expert_file_name)))
        if len(input_paths) == 0:
            return [], [], []

        for input_path in input_paths:
            data.append(load_file(input_path))
        id_data, weight_data, repr_data = zip(*data)
        id_data = torch.cat(id_data, 0)
        weight_data = torch.cat(weight_data, 0)
        repr_data = torch.cat(repr_data, 0)
        return id_data, weight_data, repr_data

    # def prune_in_diff_weights(self):
    #     for prune_weight in self.config.prune_weights_list:
    #         self._prune(prune_weight)
    #     file_path = r"{}/{}/{}/expert_original".format(self.ctx_embeddings_dir,
    #                                                    self.config.dataset,self.content_topk)
    #     shutil.rmtree(file_path)

    # def _prune(self, prune_weight):
    #     merge_out_dir = r"{}/{}/{}/expert/{}".format(self.ctx_embeddings_dir, self.config.dataset, self.content_topk, prune_weight)
    #     print(prune_weight)
    #     if not os.path.exists(merge_out_dir):
    #         os.makedirs(merge_out_dir)
    #     executor = concurrent.futures.ThreadPoolExecutor(max_workers=1000)
    #     file_path = r"{}/{}/{}/expert_original".format(self.ctx_embeddings_dir,
    #                                                    self.config.dataset, self.content_topk)
    #
    #
    #     for file in os.listdir(file_path):
    #         ctx_id, ctx_weight, ctx_repr = self.load_context_expert(file)
    #         if len(ctx_id) == 0:
    #             continue
    #         selected = torch.where(ctx_weight > float(prune_weight))
    #         ctx_id = ctx_id[selected]
    #         ctx_weight = ctx_weight[selected]
    #         ctx_repr = ctx_repr[selected]
    #         if len(ctx_id) == 0:
    #             continue
    #         path = os.path.join(merge_out_dir, file)
    #         executor.submit(CitadelIndex.save_file, (path, (ctx_id, ctx_weight, ctx_repr)))
    #     executor.shutdown()

    def save_encode(self, batch_results_list, batch_cls_list):
        def save_file(entry):
            path, output = entry
            with open(path, "wb") as f:
                pickle.dump(output, f, protocol=4)

        def prune_experts(ids, weights, reprs, prune_weight):
            if len(ids) == 0:
                return None
            selected = torch.where(weights > float(prune_weight))
            ids = ids[selected]
            weights = weights[selected]
            reprs = reprs[selected]
            if len(ids) == 0:
                return None
            return (ids, weights, reprs)

        def parallel_write(expert_dict, output_dir):
            os.makedirs(output_dir, exist_ok=True)

            for prune_weight in self.config.prune_weights_list:
                print(prune_weight)
                final_save_dir = r"{}/{}".format(output_dir, str(prune_weight))
                if not os.path.exists(final_save_dir):
                    os.makedirs(final_save_dir)
                results = []
                for k, output in tqdm(expert_dict.items()):
                    ids, weights, reprs = zip(*output)
                    ctx_ids = deepcopy(torch.LongTensor(ids))
                    ctx_weights = deepcopy(torch.stack(weights, 0).to(torch.float32))
                    ctx_reprs = deepcopy(torch.stack(reprs, 0).to(torch.float32))
                    expert_data = prune_experts(ctx_ids, ctx_weights, ctx_reprs, prune_weight)
                    if expert_data is None:
                        continue
                    else:
                        ids, weights, reprs = expert_data
                    results.append((os.path.join(final_save_dir, f"{k}.pkl"), (ids, weights, reprs)))

                with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
                    for path, output in results:
                        executor.submit(save_file, (path, output))


        save_dir = r"{}/{}/{}".format(self.ctx_embeddings_dir, self.config.dataset, self.content_topk)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        expert_embeddings = collections.defaultdict(list)
        cls_embeddings = []
        for batch_contexts_repr, batch_cls in tqdm(zip(batch_results_list, batch_cls_list)):
            if len(batch_cls) > 0:
                cls_embeddings.append(batch_cls)
            for contexts_repr in batch_contexts_repr:
                for expert_id, res in contexts_repr.items():
                    expert_embeddings[expert_id].extend(res)


        if len(cls_embeddings) > 0:
            cls_embeddings = torch.cat(cls_embeddings, 0).to(torch.float32)
            cls_out_path = os.path.join(
                self.ctx_embeddings_dir, self.config.dataset, f"cls.pkl")
            print(f"\nWriting tensors to {cls_out_path}")
            save_file((cls_out_path, cls_embeddings))

        embedding_out_dir = os.path.join(
            self.ctx_embeddings_dir, self.config.dataset, self.content_topk, f"expert")
        print(f"\nWriting tensors to {embedding_out_dir}")
        parallel_write(expert_embeddings, embedding_out_dir)  # make sure rank 0 waits

    def _load_meta_data(self):
        meta_data_path = r"{}/{}.json".format(self.config.metadata_dir, self.config.dataset)
        with open(meta_data_path, "r") as f:
            self.meta_data = json.load(f)

    @staticmethod
    def save_file(entry):
        path, output = entry
        with open(path, "wb") as f:
            pickle.dump(output, f, protocol=4)