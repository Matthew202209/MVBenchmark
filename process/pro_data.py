import json
import re

import numpy as np
import ujson
from pyarrow import float16
from tqdm import tqdm


def create_new_2_old_list(corpus_file):
    new_2_old = []
    num_lines = sum(1 for i in open(corpus_file, 'rb'))
    with open(corpus_file, encoding='utf8') as fIn:
        for line in tqdm(fIn, total=num_lines):
            line = ujson.loads(line)
            new_2_old.append(line.get('doc_id'))

    return new_2_old

def count_tokens(text):
    # 使用正则表达式分割文本为单词列表
    tokens = re.findall(r'\b\w+\b', text)
    return len(tokens)



class Corpus:
    def __init__(self, root_dir, dataset):
        self.root_dir = root_dir
        self.dataset = dataset
        self.num_docs = 0
        self.avg_num_tokens = 0
        self.total_num_tokens = 0
        self.max_num_tokens = 0
        self.min_num_tokens = 0
        self.num_tokens_list = []
        self.corpus_dict = {}

    def load_corpus(self):
        corpus_file = r"{}/corpus/{}.jsonl".format(self.root_dir, self.dataset)
        num_lines = sum(1 for i in open(corpus_file, 'rb'))
        with open(corpus_file, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus_dict[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    self.corpus_dict[line.get("doc_id")] = line.get("text")


    def cal_num_docs(self):
        self.num_docs = len(self.corpus_dict.keys())

    def cal_num_tokens(self):
        def count_tokens(text):
            # 使用正则表达式分割文本为单词列表，排除标点符号
            tokens = re.findall(r'\b[a-zA-Z]+\b', text)
            return len(tokens)
        for doc in self.corpus_dict.values():
            self.num_tokens_list.append(count_tokens(doc))

    def cal_avg_num_tokens(self):
        self.avg_num_tokens = np.mean(np.array(self.num_tokens_list))

    def cal_total_num_tokens(self):
        self.total_num_tokens = np.sum(np.array(self.num_tokens_list))

    def cal_max_num_tokens(self):
        self.max_num_tokens = np.max(np.array(self.num_tokens_list))

    def cal_min_num_tokens(self):
        self.min_num_tokens = np.min(np.array(self.num_tokens_list))

    def save_metadate(self):
        save_path = r"{}/metadata/{}.json".format(self.root_dir, self.dataset)
        metadata={
            "num_docs": int(self.num_docs),
            "avg_num_tokens" : float(self.avg_num_tokens),
            "total_num_tokens": float(self.total_num_tokens),
            "num_tokens_list": self.num_tokens_list
        }

        with open(save_path, 'w', encoding='utf-8') as json_file:
            json.dump(metadata, json_file, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    root_dir = r"/home/chunming/data/chunming/projects/MVBenchmark/data"
    dataset_list = ["antique","arguana", "clinicaltrials", "dbpedia-entity",
                    "fiqa", "lotte_pooled_dev", "msmarco-passage","nfcorpus",
                    "quora","scidocs", "scifact", "car"]
    # dataset_list = ["fiqa"]
    for data in dataset_list:
        corpus = Corpus(root_dir, data)
        corpus.load_corpus()
        corpus.cal_num_docs()
        corpus.cal_num_tokens()
        corpus.cal_avg_num_tokens()
        corpus.cal_total_num_tokens()
        corpus.save_metadate()







