import json
import os
import gzip
from tqdm import tqdm

import pandas as pd
import torch
from models.Colbert.infra import Run, RunConfig, ColBERTConfig
from models.Colbert.data import Queries, Collection
from models.Colbert import Indexer, Searcher
from time import time
import ir_datasets

checkpoint = '/home/chunming/Projects/Multivector/MVBenchmark/checkpoints/colbertv2.0'
dataset = "nfcorpus"
index_name = f'{dataset}.2bits'
topk = 20

if __name__ == '__main__':

    json_dir_root = r"/home/chunming/Projects/Multivector/MVBenchmark/data"
    query_json_dir = r"{}/query".format(json_dir_root)
    label_json_dir = r"{}/label".format(json_dir_root)
    with open(r"{}/{}.json".format(query_json_dir, dataset), 'r', encoding="utf-8") as f:
        queries = json.load(f)
    qrels = pd.read_csv(r"{}/{}.csv".format(label_json_dir, dataset))

    with Run().context(RunConfig(experiment='Colbert')):
        searcher = Searcher(checkpoint=checkpoint,index=index_name)
    # for nprobe in [1, 2, 3]:
    #     for ncandidates in [2 ** 9, 2 ** 10, 2 ** 11]:
    for nprobe in [1]:
        for ncandidates in [2 ** 9]:
            searcher.config.nprobe = nprobe
            searcher.config.ncandidates = ncandidates
            searcher.search_vanilla_colbertv2_Q(queries, k=topk)

