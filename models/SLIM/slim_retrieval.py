import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding
from typing import Dict, List, Optional, Union
from pyserini.index.lucene import IndexReader
from dataset.BenchmarkDataset import BenchmarkQueriesDataset
from models.SLIM.slim_encoder import SlimEncoder
import scipy
from pyserini.pyclass import autoclass, JFloat, JInt, JArrayList, JHashMap
JSimpleImpactSearcher = autoclass('io.anserini.search.SimpleImpactSearcher')
JScoredDoc = autoclass('io.anserini.search.ScoredDoc')


class SlimSearcher:
    def __init__(self, encoded_corpus, index_dir: str):
        self.index_dir = index_dir
        self.object = JSimpleImpactSearcher(index_dir)
        self.idf = self._compute_idf(index_dir)


        print("Loading sparse corpus vectors for fast reranking...")
        with open(os.path.join(encoded_corpus, "sparse_range.pkl"), "rb") as f:
            self.sparse_ranges = pickle.load(f)
        sparse_vecs = scipy.sparse.load_npz(os.path.join(encoded_corpus, "sparse_vec.npz"))
        self.sparse_vecs = [sparse_vecs[start:end] for start, end in tqdm(self.sparse_ranges)]

    def search(self, q: str, k: int = 10, fields=dict()) -> List[JScoredDoc]:
        jfields = JHashMap()
        for (field, boost) in fields.items():
            jfields.put(field, JFloat(boost))
    
        fusion_encoded_query, sparse_encoded_query = self.query_encoder.encode(q, return_sparse=True)
        jquery = JHashMap()
        for (token, weight) in fusion_encoded_query.items():
            if token in self.idf and self.idf[token] > self.min_idf:
                jquery.put(token, JInt(weight))
        search_k = k * (self.min_idf + 1)
        hits = self.object.search(jquery, search_k)
        hits = self.object.searchFields(jquery, jfields, search_k)
        hits = self.fast_rerank([sparse_encoded_query], {0: hits}, k)[0]
        return hits

    @staticmethod
    def _compute_idf(index_path):
 
        index_reader = IndexReader(index_path)
        tokens = []
        dfs = []
        for term in index_reader.terms():
            dfs.append(term.df)
            tokens.append(term.term)
        idfs = np.log((index_reader.stats()['documents'] / (np.array(dfs))))
        return dict(zip(tokens, idfs))

#
# searcher = SlimSearcher(args.encoded_corpus, args.index, args.encoder, args.min_idf)

class SLIMRetrieval:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.transformer_model_dir = self.config.transformer_model_dir
        self.checkpoint_path = self.config.check_point_path
        self.topk = config.content_topk

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

if __name__ == '__main__':
    from pyserini.search import LuceneImpactSearcher

    # 初始化搜索器，使用预构建的 SLIM 模型索引
    searcher = LuceneImpactSearcher.from_prebuilt_index('msmarco-v1-passage-slimr')

    # 执行检索
    hits = searcher.search('what is a lobster roll?', impact=True)
