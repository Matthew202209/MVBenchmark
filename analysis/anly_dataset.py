import json
import os

import numpy as np
import pandas as pd
import ujson
from pyarrow.dataset import dataset
from pyserini.index import IndexReader
from pyserini.index.lucene import LuceneIndexer
from tqdm import tqdm

from process.pro_data import count_tokens


class Dataset:
    def __init__(self, root_dir:str, dataset: str, threads:int):
        self.root_dir = root_dir
        self.dataset = dataset
        self.threads = threads
        self.corpus={}
        self.queries={}
        self.cf_dict_corpus = None
        self.cf_dict_queries = None
        self.df_dict_corpus = None
        self.df_dict_queries = None
        self.stats_corpus = None
        self.stats_queries = None
        self.load_corpus()
        self.load_query()
        self.load_lucene()

    def load_corpus(self):
        corpus_file = r"{}/corpus/{}.jsonl".format(self.root_dir, self.dataset)
        num_lines = sum(1 for i in open(corpus_file, 'rb'))
        with open(corpus_file, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    self.corpus[line.get("doc_id")] = line.get("text")

    def load_query(self):
        with open(r"{}/query/{}.json".format(self.root_dir, self.dataset), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)
    def load_lucene(self):
        lucene_index_corpus_save_path = r"{}/lucene/corpus/{}".format(self.root_dir, self.dataset)
        lucene_index_query_save_path = r"{}/lucene/query/{}".format(self.root_dir, self.dataset)
        if not os.path.exists(lucene_index_corpus_save_path) or not os.path.exists(lucene_index_query_save_path):
            self.build_lucene()

        self.cf_dict_corpus, self.df_dict_corpus, self.stats_corpus = Dataset.index2stats(lucene_index_corpus_save_path)
        self.cf_dict_queries, self.df_dict_queries, self.stats_queries = Dataset.index2stats(lucene_index_query_save_path)

    def build_lucene(self):
        lucene_index_corpus_save_path = r"{}/lucene/corpus/{}".format(self.root_dir, self.dataset)
        lucene_index_query_save_path = r"{}/lucene/query/{}".format(self.root_dir, self.dataset)
        lucene_corpus_data = Dataset.get_lucene_data(self.corpus)
        lucene_query_data = Dataset.get_lucene_data(self.queries)

        indexer = LuceneIndexer(lucene_index_corpus_save_path, threads=self.threads)
        indexer.add_batch_dict(lucene_corpus_data)
        indexer.close()

        indexer = LuceneIndexer(lucene_index_query_save_path, threads=self.threads)
        indexer.add_batch_dict(lucene_query_data)
        indexer.close()

    def cal_jaccard(self):
        ret = (float(len(set(self.cf_dict_queries).intersection(set(self.cf_dict_corpus)))) /
               float(len(set(self.cf_dict_queries).union(set(self.cf_dict_corpus)))))
        return ret


    def cal_weighted_jaccard(self):
        d1 = Dataset.filter_freq_dict(Dataset.cf2freq(self.cf_dict_queries))
        d2 = Dataset.filter_freq_dict(Dataset.cf2freq(self.cf_dict_corpus))

        term_union = set(d1).union(set(d2))
        min_sum = max_sum = 0
        for t in term_union:
            if t not in d1:
                max_sum += d2[t]
            elif t not in d2:
                max_sum += d1[t]
            else:
                min_sum += min(d1[t], d2[t])
                max_sum += max(d1[t], d2[t])
        ret = float(min_sum) / float(max_sum)
        return ret


    @staticmethod
    def get_lucene_data(data_dict):
        lucene_data = []
        for idx, contents in enumerate(data_dict.values()):
            if contents == "":
                continue
            else:
                lucene_data.append({"id":str(idx), "contents": contents})
        return lucene_data

    @staticmethod
    def index2stats(index_path):
        index_reader = IndexReader(index_path)

        terms = index_reader.terms()

        cf_dict = {}
        df_dict = {}
        for t in terms:
            txt = t.term
            df = t.df
            cf = t.cf
            cf_dict[txt] = int(cf)
            df_dict[txt] = int(df)

        return cf_dict, df_dict, index_reader.stats()

    @staticmethod
    def cf2freq(d):
        total = Dataset.count_total(d)
        new_d = {}
        for t in d:
            new_d[t] = float(d[t]) / float(total)
        return new_d

    @staticmethod
    def df2idf(d, n):
        total = n
        new_d = {}
        for t in d:
            new_d[t] = float(n) / float(d[t])
        return new_d

    @staticmethod
    def filter_freq_dict(freq_d, threshold=0.0001):
        new_d = {}
        for t in freq_d:
            if freq_d[t] > threshold:
                new_d[t] = freq_d[t]
        return new_d


    @staticmethod
    def count_total(d):
        s = 0
        for t in d:
            s += d[t]
        return s

class AnalysisDatasets:
    def __init__(self, root:str, save_dir:str, dataset_names: list, threads:int):
        self.root = root
        self.dataset_names = dataset_names
        self.threads = threads
        self.dataset_list = []
        self.results_df = None

    def set_datasets(self):
        for dataset_name in tqdm(self.dataset_names):
            self.dataset_list.append(Dataset(self.root, dataset_name, self.threads))


    def set_results(self):
        self.results_df = pd.DataFrame({"dataset": self.dataset_names})
    def run_statistics(self):
        self.cal_num_query_doc()
        self.cal_avg_word_lengths()
        self.cal_jaccard_similarity()
        self.results_df.to_csv(r"{}/dataset_statistics.csv".format(self.root), index=False)


    def cal_num_query_doc(self):
        self.results_df["num_corpus"] = pd.NA
        self.results_df["num_query"] = pd.NA
        self.results_df["avg_c_q"] = pd.NA
        for dataset in self.dataset_list:
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "num_corpus"] = len(list(dataset.corpus.keys()))
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "num_query"] = len(list(dataset.query.keys()))
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "avg_c_q"] = len(list(dataset.corpus.keys()))/len(list(dataset.query.keys()))


    def cal_avg_word_lengths(self):
        self.results_df["wl_corpus"] = pd.NA
        self.results_df["wl_queries"]= pd.NA
        for dataset in self.dataset_list:
            num_token_corpus = [count_tokens(corpus) for corpus in dataset.corpus.values()]
            num_token_query = [count_tokens(query) for query in dataset.query.values()]
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "wl_corpus"] = np.mean(np.array(num_token_corpus))
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "wl_queries"] = np.mean(np.array(num_token_query))

    def cal_jaccard_similarity(self):
        self.results_df["jaccard_similarity"] = pd.NA
        for dataset in self.dataset_list:
            self.results_df.loc[self.results_df["dataset"] == dataset.dataset, "jaccard_similarity"] = dataset.cal_jaccard

if __name__ == '__main__':
    root = r"/home/chunming/data/chunming/projects/MVBenchmark/data"
    save_dir = r"/home/chunming/data/chunming/projects/MVBenchmark/results/data_analysis"
    dataset_name = ["antique","arguana", "clinicaltrials", "dbpedia-entity",
                    "fiqa", "lotte_pooled_dev", "msmarco-passage","nfcorpus",
                    "quora","scidocs", "scifact"]
    threads = 10
    ad=AnalysisDatasets(root, save_dir, dataset_name, threads)
    ad.set_datasets()
    ad.run_statistics()







