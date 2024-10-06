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

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm
from models.Citadel.citadel_dataloader import CitadelDataset, split_and_load_dataset
from models.Citadel.citadel_model import CITADELEncoder
from models.Citadel.citadel_transformer import HFTransform
from models.Citadel.citadel_utils import process_check_point
# 设置要使用的 GPU


class CitadelIndex:
    def __init__(self, config):
        self.config = config
        self.transformer_model_dir = config.transformer_model_dir
        self.ctx_embeddings_dir = config.ctx_embeddings_dir
        self.latency = collections.defaultdict(float)
        self.content_topk = str(config.content_topk)
        self.num_gpus = len(config.gpus)
        self.meta_data = None
        self.encode_loader = None
        self.context_encoder = None
        self.dataset = None
        self.dataloaders= None
        self.results = []



    def save_metadata(self):
        metadata_file_path = r"{}/{}/metadata.json".format(self.ctx_embeddings_dir, self.config.dataset)
        self._load_meta_data()
        self.meta_data["index_time"] = self.latency["index_time"]

        with open(metadata_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(self.meta_data, json_file, ensure_ascii=False, indent=4)


    def save_m(self, cls_repr_list, contexts_repr_list):
        save_root = r"{}/{}".format(self.ctx_embeddings_dir, self.config.dataset)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        cls_save_dir = r"{}/{}/cls_m.jsonl".format(self.ctx_embeddings_dir, self.config.dataset)
        contexts_save_dir = r"{}/{}/context_m.jsonl".format(self.ctx_embeddings_dir, self.config.dataset)
        with open(cls_save_dir, 'w') as outfile:
            print(r"Saving m cls_repr_list......")
            # 写入数组
            for cls_repr in cls_repr_list:
                json_line = json.dumps(cls_repr.tolist())
                # 写入文件，并添加换行符
                outfile.write(json_line + '\n')

        with open(contexts_save_dir, 'w') as outfile:
            print(r"Saving m contexts_repr_list......")
            for contexts_repr in contexts_repr_list:
                json_line = json.dumps(contexts_repr)
                outfile.write(json_line + '\n')




    def _new_parallel_encode(self):


        print(self.config.gpus)
        print("Number of GPUs:", self.num_gpus)
        print("Run parallel_encode...")
        processes = []
        manager = mp.Manager()
        results = manager.list([[] for _ in self.config.gpus])
        start_time = time.time()
        for idx, dataloader in enumerate(self.dataloaders):
            print(idx)
            device = f'cuda:{self.config.gpus[idx]}'  # 分配GPU
            p = mp.Process(target=self._new_worker, args=(dataloader, device, results, idx))
            p.start()
            processes.append(p)

        for p in tqdm(processes):
            p.join()

        cls_repr_list = []
        contexts_repr_list = []
        for result in results:
            cls_repr_list.extend(result["cls_repr_list"])
            contexts_repr_list.extend(result["contexts_repr_list"])

        print("Finished encoding.")
        end_time = time.time()
        self.latency["index_time"] = end_time - start_time
        self.save_m(cls_repr_list, contexts_repr_list)
        return cls_repr_list, contexts_repr_list


    def new_save_encode(self, cls_repr_list, contexts_repr_list):
        def save_file(entry):
            path, output = entry
            with open(path, "wb") as f:
                pickle.dump(output, f, protocol=4)

        def parallel_write(expert_dict, output_dir):
            results = []
            for k, output in tqdm(expert_dict.items()):
                ids, weights, reprs = zip(*output)
                reprs = np.array(reprs)
                ids = torch.tensor(deepcopy(ids), dtype=torch.int64)
                weights = torch.tensor(deepcopy(weights), dtype=torch.float32)
                reprs = torch.tensor(deepcopy(reprs), dtype=torch.float32)
                results.append((os.path.join(output_dir, f"{k}.pkl"), (ids, weights, reprs)))

            with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
                for path, output in results:
                    executor.submit(save_file, (path, output))

        cls_repr_list = np.array(cls_repr_list)
        cls_embeddings = torch.tensor(cls_repr_list).to(torch.float32)
        root_save = os.path.join(
            self.ctx_embeddings_dir, self.config.dataset)
        if not os.path.exists(root_save):
            os.makedirs(root_save)
        cls_out_path = os.path.join(
            self.ctx_embeddings_dir, self.config.dataset, f"cls.pkl")
        print(f"\nWriting tensors to {cls_out_path}")
        save_file((cls_out_path, cls_embeddings))

        for content_topk in self.config.content_topk_list:
            for prune_weight in self.config.prune_weights_list:
                print(r"Content_topk:{}; Prune_weight:{}".format(content_topk, prune_weight))
                expert_embeddings = collections.defaultdict(list)
                for contexts_repr in tqdm(contexts_repr_list):
                    for token_idx in range(len(contexts_repr[0])):
                        corpus_id = contexts_repr[0][token_idx]
                        expert_id = np.array(contexts_repr[1][token_idx][:content_topk])
                        expert_weight = np.array(contexts_repr[2][token_idx][:content_topk])
                        expert_repr = np.array(contexts_repr[3][token_idx])
                        if np.sum(np.array(expert_weight)) == 0:
                            continue
                        for j, this_expert_id in enumerate(expert_id):
                            if expert_weight[j] <= prune_weight:
                                continue
                            expert_set = (corpus_id, expert_weight[j], expert_weight[j] * expert_repr)
                            expert_embeddings[this_expert_id].append(expert_set)
                embedding_out_dir = os.path.join(
                    self.ctx_embeddings_dir, self.config.dataset, str(content_topk), f"expert", str(prune_weight))

                if not os.path.exists(embedding_out_dir):
                    os.makedirs(embedding_out_dir)

                print(f"\nWriting tensors to {embedding_out_dir}")
                parallel_write(expert_embeddings, embedding_out_dir)  # make sure rank 0 waits

    def _parallel_encode(self):
        batch_results_list = []
        batch_cls_list = []
        print(self.config.gpus)
        print("Number of GPUs:", self.num_gpus)
        print("Run parallel_encode...")
        processes = []
        manager = mp.Manager()
        results = manager.list([[] for _ in self.config.gpus])
        start_time = time.time()
        for idx, dataloader in enumerate(self.dataloaders):
            print(idx)
            device = f'cuda:{self.config.gpus[idx]}'  # 分配GPU
            p = mp.Process(target=self._new_worker, args=(dataloader, device, results, idx))
            p.start()
            processes.append(p)

        for p in tqdm(processes):
            p.join()
            
        contexts_repr_list = []
        for result in results:
            contexts_repr_list.extend(result)
        print("Finished encoding.")
        for contexts_repr in tqdm(contexts_repr_list):
            corpus_ids = contexts_repr["corpus_ids"]
            input_ids = contexts_repr["input_ids"]
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
                        input_ids[batch_id][1:]):
                    if attention_score > 0:
                        selected = expert_topk_weights > float(self.config.weight_threshold)
                        expert_topk_weights = expert_topk_weights[selected]
                        expert_topk_ids = expert_topk_ids[selected]
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            results[expert_id.item()].append(
                                [int(corpus_id), expert_weight, expert_weight * expert_repr])
                batch_results.append(results)
            batch_results_list.append(batch_results)
            batch_cls_list.append(batch_cls)
        end_time = time.time()
        self.latency["index_time"] = end_time - start_time
        return batch_results_list, batch_cls_list


    def _prepare_parallel_data(self):
        print(self.num_gpus)
        transform = HFTransform(self.config.transformer_model_dir, self.config.max_seq_len)
        self.dataset = CitadelDataset(self.config, transform)
        self.dataloaders = split_and_load_dataset(self.dataset, num_parts=self.num_gpus,
                                              batch_size=self.config.encode_batch_size)
        
    def _new_worker(self, dataloader, device, results, idx):
        print(device)
        model = self._prepare_model_parallel(device)  # 初始化模型
        model.eval()
        contexts_repr_list = []
        result = {}
        cls_repr_list = []

        for batch in tqdm(dataloader):
            print(idx)
            corpus_ids = list(batch.data["corpus_ids"][0])
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        if k == "corpus_ids":
                            continue
                        contexts_ids_dict[k] = v.to(device)
                    batch.data = contexts_ids_dict
                    del contexts_ids_dict
                    contexts_repr = model(batch, topk=10, add_cls=True)
                    for batch_id, rep in enumerate(contexts_repr["expert_repr"]):
                        corpus_id = int(corpus_ids[batch_id])
                        attention_mask = contexts_repr["attention_mask"][batch_id]
                        expert_ids = contexts_repr["expert_ids"][batch_id]
                        expert_weights = contexts_repr["expert_weights"][batch_id]
                        # 使用masked_select选择非零元素
                        attention_mask = attention_mask.bool()
                        expert_rep = list(rep[attention_mask].detach().cpu().numpy().tolist())
                        expert_ids = list(expert_ids[attention_mask].detach().cpu().numpy().tolist())
                        expert_weights = list(expert_weights[attention_mask].detach().cpu().numpy().tolist())
                        corpus_id_list = [corpus_id for _ in range(len(expert_rep))]
                        contexts_repr_list.append((corpus_id_list, expert_ids, expert_weights, expert_rep))
                    cls_repr_list.append(contexts_repr["cls_repr"].detach().cpu().numpy())
     
        cls_repr_list = np.concatenate(cls_repr_list, axis=0)
        result["cls_repr_list"] = cls_repr_list
        result["contexts_repr_list"] = contexts_repr_list
        results[idx] = result
    
    
    def _worker(self, dataloader, device, results, idx):
        print(device)
        model = self._prepare_model_parallel(device)  # 初始化模型
        model.eval()
        contexts_repr_list = []
        for batch in tqdm(dataloader):
            print(idx)
            corpus_ids = list(batch.data["corpus_ids"][0])
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        if k == "corpus_ids":
                            continue
                        contexts_ids_dict[k] = v.to(device)
                    batch.data = contexts_ids_dict
                    del contexts_ids_dict
                    contexts_repr = model(batch, topk=int(self.content_topk), add_cls=True)
                    contexts_repr["input_ids"] = batch.data["input_ids"]
            contexts_repr = {k: v.detach().cpu().numpy() for k, v in contexts_repr.items()}
            contexts_repr["corpus_ids"] = corpus_ids 
            contexts_repr_list.append(contexts_repr)
        results[idx] += contexts_repr_list



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
    def _prepare_model_parallel(self, device):
        context_encoder = CITADELEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)
        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]

        checkpoint_dict = process_check_point(checkpoint_dict)
        context_encoder.load_state_dict(checkpoint_dict)
        return context_encoder.to(device)

    def _prepare_model(self):

        self.context_encoder = CITADELEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout,
                                              tok_projection_dim=self.config.tok_projection_dim,
                                              cls_projection_dim=self.config.cls_projection_dim)

        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]

        checkpoint_dict = process_check_point(checkpoint_dict)
        self.context_encoder.load_state_dict(checkpoint_dict)

        if self.config.is_parallel:
            self.context_encoder = nn.DataParallel(self.context_encoder.to(self.config.device))
        else:
            self.context_encoder.to(self.config.device)

    def _encode(self):
        batch_results_list = []
        batch_cls_list = []
        start_time = time.time()
        print("Encoding data...")
        contexts_repr_list = []
        print("Model device:", next(self.context_encoder.parameters()).device)
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
                    contexts_repr["corpus_ids"] = corpus_ids
                    contexts_repr["input_ids"]= batch.data["input_ids"]
     
            contexts_repr_list.append(contexts_repr)
        
        print("Finished encoding.")

        for contexts_repr in tqdm(contexts_repr_list):
            corpus_ids = contexts_repr["corpus_ids"]
            input_ids = contexts_repr["input_ids"]
            contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items() if k != "corpus_ids" or k != "input_ids"}
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
                        input_ids[batch_id][1:]):
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

    def parallel_encode(self):
        def process_contexts_repr(contexts_repr_list_segment):
            batch_results_list = []
            batch_cls_list = []
            for contexts_repr in tqdm(contexts_repr_list_segment):
                corpus_ids = contexts_repr["corpus_ids"]
                input_ids = contexts_repr["input_ids"]
                contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items() if k != "corpus_ids"}
                batch_results = []

                batch_cls = contexts_repr["cls_repr"]
                for batch_id, corpus_id in enumerate(corpus_ids):
                    results = collections.defaultdict(list)
                    for expert_repr, expert_topk_ids, expert_topk_weights, attention_score, context_id in zip(
                            contexts_repr["expert_repr"][batch_id],
                            contexts_repr["expert_ids"][batch_id],
                            contexts_repr["expert_weights"][batch_id],
                            contexts_repr["attention_mask"][batch_id],
                            input_ids[batch_id][1:]):
                        if attention_score > 0:
                            if len(contexts_repr["expert_ids"].shape) == 2:  # COIL and ColBERT
                                if expert_topk_weights > 0:
                                    results[expert_topk_ids.item()].append(
                                        [int(corpus_id), expert_topk_weights, expert_topk_weights * expert_repr])
                            else:  # CITADEL

                                selected = torch.where(expert_topk_weights > float(self.config.weight_threshold))
                                expert_topk_weights = expert_topk_weights[selected]
                                expert_topk_ids = expert_topk_ids[selected]
                                for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                                    results[expert_id.item()].append(
                                        [int(corpus_id), expert_weight, expert_weight * expert_repr])
                    batch_results.append(results)
                batch_results_list.append(batch_results)
                batch_cls_list.append(batch_cls)
            return batch_results_list, batch_cls_list

        final_batch_results_list = []
        final_batch_cls_list = []

        start_time = time.time()
        print("Encoding data...")
        contexts_repr_list = []
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
                    contexts_repr["corpus_ids"] = corpus_ids
                    contexts_repr["input_ids"] = batch.data["input_ids"]

            contexts_repr_list.append(contexts_repr)

        print("Finished encoding.")

        batch_size = len(contexts_repr_list) // self.config.num_threads
        with concurrent.futures.ThreadPoolExecutor(max_workers= self.config.num_threads) as executor:
            futures = []

            # 划分列表为多个片段并提交任务
            for i in range(self.config.num_threads):
                segment = contexts_repr_list[
                          i * batch_size:(i + 1) * batch_size] if i <  self.config.num_threads - 1 else contexts_repr_list[
                                                                                           i * batch_size:]
                futures.append(executor.submit(process_contexts_repr, segment))

            # 收集结果
            for future in concurrent.futures.as_completed(futures):
                batch_results, batch_cls = future.result()
                final_batch_results_list.extend(batch_results)
                final_batch_cls_list.extend(batch_cls)

        end_time = time.time()
        self.latency["index_time"] = end_time - start_time
        return final_batch_results_list, final_batch_cls_list
    def setup(self):
        self._prepare_data()
        self._prepare_model()

    def run(self):
        batch_results_list, batch_cls_list = self._encode()
        self.save_encode(batch_results_list, batch_cls_list)
        self.save_metadata()

    def parallel_setup(self):
        self._prepare_parallel_data()

    def parallel_run(self):
        self.parallel_setup()
        cls_repr_list, contexts_repr_list = self._new_parallel_encode()
        self.new_save_encode(cls_repr_list, contexts_repr_list)
        self.save_metadata()

    # def parallel_run(self):
    #     self.parallel_setup()
    #     batch_results_list, batch_cls_list = self._new_parallel_encode()
    #     self.new_save_encode(batch_results_list, batch_cls_list)
    #     self.save_metadata()

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
                    ctx_ids = deepcopy(torch.stack(ids, 0).to(torch.float32))
                    
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
                batch_cls = torch.tensor(batch_cls).to(torch.float32).cpu()
                cls_embeddings.append(batch_cls)
            for contexts_repr in batch_contexts_repr:
                for expert_id, res in contexts_repr.items():
                    new_res = []
                    for r in res:
                        re = [torch.tensor(item).to(torch.float32).cpu() for item in r]
                        new_res.append(re)
                    expert_embeddings[expert_id].extend(new_res)


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



