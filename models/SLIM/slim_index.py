import glob
import os
import jsonlines
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, BertTokenizer

from dataset.BenchmarkDataset import BenchmarkDataset
from models.SLIM.slim_encoder import SlimEncoder


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
            num_workers=self.config.dataloader_num_workers,
        )


    def _prepare_model(self):
        self.context_encoder = SlimEncoder(model_path=self.config.transformer_model_dir,
                                              dropout=self.config.dropout)

        checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]

        # checkpoint_dict = process_check_point(checkpoint_dict)

        self.context_encoder.load_state_dict(checkpoint_dict)
        self.context_encoder.to(self.config.device)

    def _encode(self):
        batch_results_list = []
        batch_sparse_vecs_list = []
        for batch in tqdm(self.encode_loader):
            contexts_ids_dict = {}
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for k, v in batch.items():
                        contexts_ids_dict[k] = v.to(self.config.device)
                batch.data = contexts_ids_dict
                del contexts_ids_dict
            contexts_repr = self.context_encoder(batch, topk=self.config.context_top_k)

            batch_sparse_vecs = []
            lengths = contexts_repr["attention_mask"].sum(1)
            for i, length in enumerate(lengths):
                batch_sparse_vecs.append(contexts_repr["sparse_weights"][i][:length, :].to_sparse().detach().cpu())

            contexts_repr = {k: v.detach().cpu() for k, v in contexts_repr.items() if k != "sparse_weights"}
            batch_results = []
            for batch_id, corpus_id in enumerate(batch["corpus_ids"]):
                results = {"id": str(corpus_id), "contents": "", "vector": {}}
                for position, (expert_topk_ids, expert_topk_weights, attention_score, context_id) in enumerate(
                        zip(contexts_repr["expert_ids"][batch_id],
                            contexts_repr["expert_weights"][batch_id],
                            contexts_repr["attention_mask"][batch_id],
                            batch.data["input_ids"].cpu()[batch_id][1:])):
                    if attention_score > 0:
                        for expert_id, expert_weight in zip(expert_topk_ids, expert_topk_weights):
                            if expert_weight > self.weight_threshold:
                                term = self.vocab[expert_id.item()]
                                tf = int(expert_weight.item() * 100)
                                results["vector"][term] = max(tf, results["vector"].get(term, 0))
                batch_results.append(results)
            batch_sparse_vecs_list.append(batch_sparse_vecs)
            batch_results_list.append(batch_results)
        self.save_encode(zip(batch_results_list, batch_sparse_vecs_list))

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