import json
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
from models.SLIM.slim_encoder import SlimEncoder, SlimQueryEncoder
import scipy
from pyserini.pyclass import autoclass, JFloat, JInt, JArrayList, JHashMap
from collections import namedtuple
JSimpleImpactSearcher = autoclass('io.anserini.search.SimpleImpactSearcher')
JScoredDoc = autoclass('io.anserini.search.ScoredDoc')


class SlimSearcher:
    def __init__(self, encoded_corpus, index_dir: str, query_encoder, min_idf=3):
        self.query_encoder = query_encoder
        self.index_dir = index_dir
        self.object = JSimpleImpactSearcher(index_dir)
        self.idf = self._compute_idf(index_dir)
        self.min_idf = min_idf


        print("Loading sparse corpus vectors for fast reranking...")
        with open(os.path.join(encoded_corpus, "sparse_range.pkl"), "rb") as f:
            self.sparse_ranges = pickle.load(f)
        sparse_vecs = scipy.sparse.load_npz(os.path.join(encoded_corpus, "sparse_vec.npz"))
        self.sparse_vecs = [sparse_vecs[start:end] for start, end in tqdm(self.sparse_ranges)]

    def search(self, q: str, k: int = 10) -> List[JScoredDoc]:
        fusion_encoded_query, sparse_encoded_query = self.query_encoder.encode(q, return_sparse=True)
        jquery = JHashMap()
        for (token, weight) in fusion_encoded_query.items():
            if token in self.idf and self.idf[token] > self.min_idf:
                jquery.put(token, JInt(weight))
        search_k = k * (self.min_idf + 1)
        hits = self.object.search(jquery, search_k)
        hits = self.fast_rerank([sparse_encoded_query], {0: hits}, k)[0]
        return hits

    def fast_rerank(self, q_embeds, results, k):
        all_scores = []
        all_docids = []
        all_q_embeds = []
        all_d_embeds = []
        all_d_lens = []
        qids = []
        for qid in results.keys():
            all_q_embeds.append(q_embeds[qid])
            qids.append(qid)
            hits = results[qid]
            docids = []
            scores = []
            d_embeds = []
            d_lens = []
            for hit in hits:
                docids.append(hit.docid)
                scores.append(hit.score)
                start, end = self.sparse_ranges[int(hit.docid)]
                d_embeds.append(self.sparse_vecs[int(hit.docid)])
                d_lens.append(end-start)
            all_scores.append(scores)
            all_docids.append(docids)
            all_d_embeds.append(d_embeds)
            all_d_lens.append(d_lens)

        entries = list(zip(all_q_embeds, all_d_embeds, all_d_lens, qids, all_scores, all_docids))
        results = [maxsim(entry) for entry in entries]
        anserini_results = {}
        for qid, scores, docids in results:
            hits = []
            for score, docid in list(zip(scores, docids))[:k]:
                hits.append(SlimResult(docid, score))
            anserini_results[qid] = hits
        return anserini_results


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
    def __init__(self, config):
        self.config = config
        self.lucene_index_dir = r"{}/Slim/{}/lucene_index".format(self.config.index_dir, self.config.dataset)
        self.encoded_corpus = r"{}/Slim/{}".format(self.config.index_dir, self.config.dataset)
        self.transformer_model_dir = self.config.transformer_model_dir
        self.checkpoint_path = self.config.check_point_path
        self.query_dict = {}
        self.query_encoder =None
        self.searcher = None
        self.topk = config.topk
    def set_up(self):
        self._prepare_data()
        self.set_query_encoder()
        self.searcher = SlimSearcher(self.encoded_corpus, self.lucene_index_dir, self.query_encoder)

    def set_query_encoder(self):
        self.query_encoder = SlimQueryEncoder(self.config.check_point_path, self.config.transformer_model_dir, device=self.config.device)
    # def _load_checkpoint(self):
    #     checkpoint_dict = torch.load(self.config.check_point_path)["state_dict"]
    #     return checkpoint_dict


    # def _set_up_model(self, checkpoint_dict):
    #     self.context_encoder = SlimEncoder(model_path=self.config.transformer_model_dir,
    #                                        dropout=self.config.dropout)
    #     self.context_encoder.load_state_dict(checkpoint_dict)
    #     self.context_encoder.to(self.config.device)

    def _prepare_data(self):
        with open(r"{}/{}.json".format(self.config.queries_dir, self.config.dataset), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)

    def run(self):
        for q_id, query in tqdm(self.queries.items()):
            self.searcher.search(query, k=self.topk)


    # def setup(self):
    #     checkpoint_dict = self._load_checkpoint()
    #     self._set_up_model(checkpoint_dict)
    #     self._sep_up_index()


SlimResult = namedtuple("SlimResult", "docid score")


def maxsim(entry):
    q_embed, d_embeds, d_lens, qid, scores, docids = entry
    if len(d_embeds) == 0:
        return qid, scores, docids
    d_embeds = scipy.sparse.vstack(d_embeds).transpose() # (LD x 1000) x D
    max_scores = (q_embed@d_embeds).todense() # LQ x (LD x 1000)
    scores = []
    start = 0
    for d_len in d_lens:
        scores.append(max_scores[:, start:start+d_len].max(1).sum())
        start += d_len
    scores, docids = list(zip(*sorted(list(zip(scores, docids)), key=lambda x: -x[0])))
    return qid, scores, docids



if __name__ == '__main__':
    from pyserini.search import LuceneImpactSearcher

    # 初始化搜索器，使用预构建的 SLIM 模型索引
    searcher = LuceneImpactSearcher.from_prebuilt_index('msmarco-v1-passage-slimr')

    # 执行检索
    hits = searcher.search('what is a lobster roll?', impact=True)
