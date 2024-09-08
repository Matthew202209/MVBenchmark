import json
import os
import gzip

import ir_measures
from ir_measures import *
from pyterrier_adaptive import CorpusGraph
from pyterrier_pisa import PisaIndex
from tqdm import tqdm

import pandas as pd
import torch
from models.Colbert.infra import Run, RunConfig, ColBERTConfig
from models.Colbert.data import Queries, Collection
from models.Colbert import Indexer, Searcher, LadrSearcher
from time import time
import ir_datasets

from process.pro_data import create_new_2_old_list
from utils.utils_memory import memory_usage

checkpoint = '/home/chunming/Projects/Multivector/MVBenchmark/checkpoints/colbertv2.0'
dataset = "nfcorpus"
method = "ladr"
topk = 30
measure = [nDCG@10, RR@10, Success@10]

if __name__ == '__main__':
    dataset_list = ["nfcorpus", "quora", "scifact", "scidocs", "antique", "arguana", "clinicaltrials"]
    for dataset in dataset_list:
        index_name = f'{dataset}.2bits'
        json_dir_root = r"{}/data".format(os.getcwd())
        save_dir = r"{}/results/{}/{}".format(os.getcwd(),method,dataset)
        perf_path = r"{}/{}".format(save_dir, "perf_results")
        rank_path = r"{}/{}".format(save_dir, "rank_results")
        eval_results_dir = r"{}/{}".format(save_dir, "eval_results")
        if not os.path.exists(perf_path):
            os.makedirs(perf_path)
        if not os.path.exists(rank_path):
            os.makedirs(rank_path)
        if not os.path.exists(eval_results_dir):
            os.makedirs(eval_results_dir)

        query_json_dir = r"{}/query".format(json_dir_root)
        label_json_dir = r"{}/label".format(json_dir_root)
        corpus_file = r"{}/corpus/{}.jsonl".format(json_dir_root, dataset)
        with open(r"{}/{}.json".format(query_json_dir, dataset), 'r', encoding="utf-8") as f:
            queries = json.load(f)
        qrels = pd.read_csv(r"{}/{}.csv".format(label_json_dir, dataset))
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["doc_id"] = qrels["doc_id"].astype(str)

        before_memory = memory_usage()
        index = PisaIndex(f'/home/chunming/Projects/Multivector/MVBenchmark/index/Ladr/{dataset}/{dataset}-n-pisa')
        bm25 = index.bm25(num_results=1000)  # number of nearest neighbours
        graph = CorpusGraph.load(
            f'/home/chunming/Projects/Multivector/MVBenchmark/index/Ladr/{dataset}/{dataset}.gbm25.128')
        after_memory = memory_usage()
        sparse_index_memory = after_memory - before_memory
        with Run().context(RunConfig(experiment='Colbert')):
            searcher = LadrSearcher(checkpoint=checkpoint, index=index_name, first_pass=bm25, corpus_graph=graph, proactive_steps=0)

        new2old = create_new_2_old_list(corpus_file)
        eval_list = []
        for num_results in [100, 500, 1000]:
            for num_neighbours in [64, 128]:
                for depth in [10, 20, 50]:
                    searcher.first_pass = index.bm25(num_results=num_results)
                    searcher.adaptive_depth = depth
                    searcher.graph = graph.to_limit_k(num_neighbours)
                    path = f'{rank_path}/ladr.num_results-{num_results}.num_neighbours-{num_neighbours}.depth-{depth}.run.gz'

                    results, perf_df = searcher.search_ladr_Q(queries, k=topk)
                    res_dict = results.todict()
                    perf_df.to_csv(r"{}/ladr.num_results-{}.num_neighbours-{}.depth-{}_perf.csv".format(perf_path,
                                                                                                        num_results,
                                                                                                        num_neighbours,
                                                                                                        depth), index=False)
                    with gzip.open(path, 'wt') as fout:
                        for qid in res_dict.keys():
                            for did, rank, score in res_dict[qid]:
                                fout.write(f'{qid} 0 {did} {rank} {score} run\n')
                    del results, perf_df
                    ranks_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
                    for i, r in ranks_results_pd.iterrows():
                        ranks_results_pd.at[i, "doc_id"] = new2old[int(r["doc_id"])]
                    eval_results = ir_measures.calc_aggregate(measure, qrels, ranks_results_pd)
                    eval_results["parameter"] = (num_results, num_neighbours, depth)
                    eval_results["num_results"] = num_results
                    eval_results["num_neighbours"] = num_neighbours
                    eval_results["depth"] = depth
                    eval_results["index_memory"] = searcher.index_memory + sparse_index_memory
                    eval_results["index_dlen"] =  len(new2old)
                    eval_list.append(eval_results)

        eval_df = pd.DataFrame(eval_list)
        eval_df.to_csv(r"{}/eval.csv".format(eval_results_dir), index=False)

