import glob
import os
import pickle
import scipy
import jsonlines
import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, BertTokenizer, AutoModelForMaskedLM

from dataset.BenchmarkDataset import BenchmarkDataset
from models.SLIM.slim_encoder import SlimEncoder
from models.SLIM.slim_utils import process_check_point


class SlimIndex:
    def __init__(self, config):
        self.config = config
        self.transformer_model_dir = config.transformer_model_dir
        self.ctx_embeddings_dir = config.ctx_embeddings_dir
        self.weight_threshold = config.weight_threshold
        self.vocab_file = config.vocab_file
        self.encode_loader = None
        self.context_encoder = None
        self.dataset = None

    def setup(self):
        self._prepare_data()
        self._prepare_model()
        with open(self.vocab_file) as f:
            lines = f.readlines()
        self.vocab = []
        for line in lines:
            self.vocab.append(line.strip())

    def run(self):
        self._encode()


    def _prepare_data(self):
        tokenizer = BertTokenizer.from_pretrained(self.transformer_model_dir, use_fast=False)

        self.dataset = BenchmarkDataset(self.config, tokenizer)
        self.dataset.load_corpus()

        self.encode_loader = DataLoader(
            self.dataset,
            batch_size=self.config.encode_batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=self.config.max_seq_len,
                padding='max_length'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=1,
        )


    def _prepare_model(self):
        self.context_encoder = SlimEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout)
        checkpoint_dict = AutoModelForMaskedLM.from_pretrained(self.config.check_point_path).state_dict()

        checkpoint_dict = process_check_point(checkpoint_dict)

        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _encode(self):
        batch_results_list = []
        # batch_sparse_vecs_list = []
        for set_id, batch in enumerate(tqdm(self.encode_loader)):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        if k == "corpus_ids":
                            corpus_ids = v
                            continue
                        contexts_ids_dict[k] = v.to(self.config.device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict
            contexts_repr = self.context_encoder(batch, topk=self.config.context_top_k)
            contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items()}
            # contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items() if k != "sparse_weights"}

            # take up corpus_ids and input_ids
            input_ids = batch.data["input_ids"].cpu()
            del batch

            sparse_weights = contexts_repr["sparse_weights"]
            expert_ids = contexts_repr["expert_ids"]
            expert_weights = contexts_repr["expert_weights"]
            attention_mask = contexts_repr["attention_mask"]
            del contexts_repr


            batch_sparse_vecs = []
            lengths = attention_mask.sum(1)
            for i, length in enumerate(lengths):
                batch_sparse_vecs.append(sparse_weights[i][:length, :].to_sparse())
            self.save_sparse_vecs(batch_sparse_vecs, set_id)
            del sparse_weights, batch_sparse_vecs

            batch_results = []
            for batch_id, corpus_id in enumerate(corpus_ids):
                results = {"id": str(corpus_id), "contents": "", "vector": {}}
                for position, (expert_topk_ids, expert_topk_weights, attention_score, context_id) in enumerate(
                        # zip(expert_ids[batch_id],
                        #     expert_weights[batch_id],
                        #     attention_mask[batch_id],
                        #     input_ids.cpu()[batch_id][1:])):
                        zip(expert_ids[batch_id],
                            expert_weights[batch_id],
                            attention_mask[batch_id],
                            input_ids[batch_id][1:])):
                    if attention_score > 0:
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            if expert_weight > self.weight_threshold:
                                term = self.vocab[expert_id.item()]
                                tf = int(expert_weight.item() * 100)
                                results["vector"][term] = max(tf, results["vector"].get(term, 0))
                batch_results.append(results)
                del results
            self.save_dense_vecs(batch_results, set_id)
            del batch_results


    def _prune(self, prune_threshold):
        prune_out_dir = r"{}/{}/expert/{}".format(self.ctx_embeddings_dir, self.config.dataset, prune_threshold)
        file_path = r"{}/{}/doc".format(self.ctx_embeddings_dir,
                                              self.config.dataset)

        if not os.path.exists(prune_out_dir):
            os.makedirs(prune_out_dir)

        with open(self.vocab_file) as f:
            lines = f.readlines()
        vocab = []
        for line in lines:
            vocab.append(line.strip())

        input_paths = glob.glob(file_path)
        for i, input_path in tqdm(enumerate(list(input_paths))):
            results = []
            with jsonlines.open(input_path) as f:
                for entry in tqdm(f):
                    vector = {}
                    for term, weight in entry["vector"].items():
                        if weight > int(float(prune_threshold) * 100):
                            vector[term] = weight
                    entry["vector"] = vector
                    entry["contents"] = ""
                    if len(vector) > 0:
                        results.append(entry)
            with jsonlines.open(f'{prune_out_dir}/context_embedding.jsonl', 'w') as writer:
                writer.write_all(results)


    def _compress(self, threshold=0.0):
        file_path = r"{}/{}/tok".format(self.ctx_embeddings_dir,
                                              self.config.dataset)
        output_dir = r"{}/{}/expert".format(self.ctx_embeddings_dir, self.config.dataset)
        input_paths = sorted(glob.glob(file_path))
        total_sparse_vecs = []
        sparse_ranges = []
        start = 0
        for i, input_path in tqdm(list(enumerate(input_paths))):
            data = torch.load(input_path)
            for sparse_vec in data:
                end = start + sparse_vec.shape[0]
                sparse_ranges.append((start, end))
                start = end
            sparse_vecs = torch.cat(data, 0).coalesce()
            indices = sparse_vecs.indices().numpy()
            values = sparse_vecs.values().numpy()
            pos = np.where(values >= threshold)[0]
            values = values[pos]
            indices = (indices[0][pos], indices[1][pos])
            sparse_vecs = csr_matrix((values, indices), shape=sparse_vecs.shape)
            total_sparse_vecs.append(sparse_vecs)
        total_sparse_vecs = vstack(total_sparse_vecs)

        output_path = f'{output_dir}/sparse_vec.npz'
        print(f"Writing tensor to {output_path}")
        scipy.sparse.save_npz(output_path, total_sparse_vecs)

        output_path = f'{output_dir}/sparse_range.pkl'
        print(f"Writing tensor to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(sparse_ranges, f)

    def save_sparse_vecs(self, sparse_vecs, set_id):
        tok_dir = r"{}/{}/tok".format(self.ctx_embeddings_dir, self.config.dataset)
        os.makedirs(tok_dir, exist_ok=True)
        sparse_embedding_path = os.path.join(
            tok_dir, f"sparse_embedding_{set_id}.pt")
        torch.save(sparse_vecs, sparse_embedding_path)


    def save_dense_vecs(self, dense_vecs, set_id):
        doc_dir = r"{}/{}/doc".format(self.ctx_embeddings_dir, self.config.dataset)
        os.makedirs(doc_dir, exist_ok=True)
        embedding_path = os.path.join(
            doc_dir, f"context_embedding_{set_id}.jsonl")
        with jsonlines.open(embedding_path, 'w') as writer:
            writer.write_all(dense_vecs)




    def save_encode(self, contexts_reprs):
        results = []
        sparse_vecs = []
        for context_repr, sparse_vec in contexts_reprs:
            results.extend(context_repr)
            sparse_vecs.extend(sparse_vec)

        doc_dir = os.path.join(self.ctx_embeddings_dir, "doc")
        tok_dir = os.path.join(self.ctx_embeddings_dir, "tok")
        os.makedirs(doc_dir, exist_ok=True)
        os.makedirs(tok_dir, exist_ok=True)
        embedding_path = os.path.join(
            doc_dir, f"context_embedding.jsonl")
        sparse_embedding_path = os.path.join(
            tok_dir, f"sparse_embedding.pt")

        print(f"\nWriting tensors to {embedding_path}")
        with jsonlines.open(embedding_path, 'w') as writer:
            writer.write_all(results)
        print(f"\nWriting tensors to {sparse_embedding_path}")
        torch.save(sparse_vecs, sparse_embedding_path)
        torch.distributed.barrier()  # make sure rank 0 waits for all to complete