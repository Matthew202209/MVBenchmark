import json
import os
import gzip

import ir_measures
from ir_measures import *
from tqdm import tqdm

import pandas as pd
import torch
from models.Colbert.infra import Run, RunConfig, ColBERTConfig
from models.Colbert.data import Queries, Collection
from models.Colbert import Indexer, Searcher
from time import time
import ir_datasets

from process.pro_data import create_new_2_old_list
from utils.utils_memory import memory_usage

checkpoint = '/home/chunming/Projects/Multivector/MVBenchmark/checkpoints/colbertv2.0'
dataset = "nfcorpus"
index_name = f'{dataset}.2bits'
topk = 30
measure = [nDCG@10, RR@10, Success@10]

if __name__ == '__main__':

    json_dir_root = r"{}/data".format(os.getcwd())
    save_dir = r"{}/results/{}".format(os.getcwd(),dataset)
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

    with Run().context(RunConfig(experiment='Colbert')):

        searcher = Searcher(checkpoint=checkpoint,index=index_name, is_colbertv2=False)


    new2old = create_new_2_old_list(corpus_file)
    eval_list = []
    for ncells in [2, 3, 4]:
        for centroid_score_threshold in [0.3, 0.4, 0.5]:
            for ndocs in [2 ** 9, 2 ** 10, 2 ** 11]:
                path = f'{rank_path}/plaid.ncells-{ncells}.cst-{centroid_score_threshold}.ndocs-{ndocs}.run.gz'
                searcher.config.ncells = ncells
                searcher.config.centroid_score_threshold = centroid_score_threshold
                searcher.config.ndocs = ndocs
                results, perf_df = searcher.search_plaid_Q(queries, k=topk)
                res_dict = results.todict()
                perf_df.to_csv(r"{}/plaid.ncells-{}.cst-{}.ndocs-{}_perf.csv".format(perf_path,
                                                                                     ncells,
                                                                                     centroid_score_threshold,
                                                                                     ndocs), index=False)
                with gzip.open(path, 'wt') as fout:
                    for qid in res_dict.keys():
                        for did, rank, score in res_dict[qid]:
                            fout.write(f'{qid} 0 {did} {rank} {score} run\n')
                del results, perf_df
                ranks_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))
                for i, r in ranks_results_pd.iterrows():
                    ranks_results_pd.at[i, "doc_id"] = new2old[int(r["doc_id"])]
                eval_results = ir_measures.calc_aggregate(measure, qrels, ranks_results_pd)
                eval_results["parameter"] = (ncells, centroid_score_threshold, ndocs)
                eval_results["ncells"] = ncells
                eval_results["centroid_score_threshold"] = centroid_score_threshold
                eval_results["ndocs"] = ndocs
                eval_results["index_memory"] = searcher.index_memory
                eval_list.append(eval_results)

    eval_df = pd.DataFrame(eval_list)
    eval_df.to_csv(r"{}/eval.csv".format(eval_results_dir), index=False)

