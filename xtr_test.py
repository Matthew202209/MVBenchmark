"""
This file is used to test the code of xtr
"""
import argparse
import gzip
import json
import os
import logging
import ir_measures
from ir_measures import *

import pandas as pd
import perf_event
import torch
from tqdm import tqdm

from models.XTR import XtrRetriever

from process.pro_data import create_new_2_old_list
from utils.utils_general import create_this_perf
from utils.utils_memory import memory_usage, get_folder_size

measure = [nDCG@10, RR@10, Success@10]

if __name__ == '__main__':

    device = "cpu"
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./checkpoints/xtr-base-en")
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--data_set_path", type=str, default='irds:beir/nfcorpus/test')
    parser.add_argument("--queries_path", type=str, default='beir/nfcorpus/test')
    parser.add_argument("--use_faiss", type=bool, default=True)
    parser.add_argument("--doc_sample_ratio", type=float, default=0.2)
    parser.add_argument("--vec_sample_ratio", type=float, default=0.2)
    parser.add_argument("--code_size", type=int, default=64)
    parser.add_argument("--nprobe", type=int, default=128)
    parser.add_argument("--token_top_k", type=int, default=8000)
    parser.add_argument("--trec_top_k", type=int, default=10)
    parser.add_argument("--dataset_dir", type=str, default="../datasets")
    parser.add_argument("--index_dir", type=str, default="index/XTR")
    parser.add_argument("--results_dir", type=str, default="./results/xtr")
    parser.add_argument("--load_index", type=bool, default=True)
    args = parser.parse_args()
    dataset_list = ["lotte_pooled_dev"]
    for dataset in dataset_list:
        args.dataset = dataset
        perf_path = r"{}/{}/{}".format(args.results_dir, args.dataset, "perf_results")
        rank_path = r"{}/{}/{}".format(args.results_dir, args.dataset, "rank_results")
        eval_results_dir = r"{}/{}/{}".format(args.results_dir, args.dataset, "eval_results")
        index_dir = r"{}/{}".format(args.index_dir, args.dataset)
        index_memory = get_folder_size(index_dir)

        if not os.path.exists(perf_path):
            os.makedirs(perf_path)
        if not os.path.exists(rank_path):
            os.makedirs(rank_path)
        if not os.path.exists(eval_results_dir):
            os.makedirs(eval_results_dir)

        json_dir_root = r"{}/data".format(os.getcwd())
        query_json_dir = r"{}/query".format(json_dir_root)
        label_json_dir = r"{}/label".format(json_dir_root)
        corpus_file = r"{}/corpus/{}.jsonl".format(json_dir_root, args.dataset)
        with open(r"{}/{}.json".format(query_json_dir, args.dataset), 'r', encoding="utf-8") as f:
            queries = json.load(f)
        ######################################
        print("Step 1 - Load XTR Retriever")
        ######################################
        print(args.dataset)
        xtr = XtrRetriever(
            model_name_or_path=args.model_name_or_path,
            use_faiss=args.use_faiss,
            device=device
        )

        ######################################
        print("Step 2 - Load Datasets")
        ######################################

        new_2_old_corpus = create_new_2_old_list(corpus_file)
        new_2_old_queries = list(queries.keys())

        # For Scifact + XTR-base-en (P100), this should take about 3 minutes.
        index_dir = f"{args.index_dir}/{args.dataset}"
        index_num = xtr.load_index(
            index_dir=index_dir,
            code_size=args.code_size,
            nprobe=args.nprobe
        )
        ######################################
        print("Step 4 - Run BEIR Evaluation")
        ######################################
        qrels = pd.read_csv(r"{}/{}.csv".format(label_json_dir, args.dataset))
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        eval_list = []
        # For Scifact, XTR-base-en (P100), this should take about 2 minutes.
        for token_top_k in [2**5,2**6,2**7,2**8,2**9]:
            TOKEN_TOP_K = token_top_k
            TREC_TOP_K = args.trec_top_k
            res_dict = {}
            # Running evaluation per query for a better latency measurement.
            all_perf =[]
            perf_encode = perf_event.PerfEvent()
            perf_retrieve = perf_event.PerfEvent()
            for q_idx, (_, query) in tqdm(enumerate(queries.items()), total=len(queries)):
                ranking, metadata, perf_encode, perf_retrieve = xtr.retrieve_docs(
                    [query],
                    perf_encode,
                    perf_retrieve,
                    token_top_k=TOKEN_TOP_K,
                    document_top_k=TREC_TOP_K,
                    return_text=False
                )
                this_perf = create_this_perf(perf_encode, perf_retrieve)
                all_perf.append(this_perf)
                res_dict[q_idx] = ranking[0]
            columns = ["encode_cycles", "encode_instructions",
               "encode_L1_misses", "encode_LLC_misses",
               "encode_L1_accesses", "encode_LLC_accesses",
               "encode_branch_misses", "encode_task_clock",
               "retrieval_cycles", "retrieval_instructions",
               "retrieval_L1_misses", "retrieval_LLC_misses",
               "retrieval_L1_accesses", "retrieval_LLC_accesses",
               "retrieval_branch_misses", "retrieval_task_clock"]
            perf_df = pd.DataFrame(all_perf, columns=columns)
            perf_df.to_csv(r"{}/xtr.token_top_k-{}.top_perf.csv".format(perf_path, TOKEN_TOP_K, "xtr"), index=False)
            path = f'{rank_path}/xtr.token_top_k-{TOKEN_TOP_K}.run.gz'
            with gzip.open(path, 'wt') as fout:
                for qid in res_dict.keys():
                    for did, rank, score in res_dict[qid]:
                        fout.write(f'{qid} 0 {did} {rank} {score} run\n')

            ranks_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))

            for i, r in ranks_results_pd.iterrows():
                ranks_results_pd.at[i, "doc_id"] = new_2_old_corpus[int(r["doc_id"])]
                ranks_results_pd.at[i, "query_id"] = new_2_old_queries[int(r["query_id"])]
            eval_results = ir_measures.calc_aggregate(measure, qrels, ranks_results_pd)
            eval_results["parameter"] = (str(token_top_k))
            eval_results["token_top_k"] = token_top_k
            eval_results["index_memory"] = index_memory
            eval_results['index_dlen'] = len(new_2_old_corpus)
            eval_list.append(eval_results)

        eval_df = pd.DataFrame(eval_list)
        eval_df.to_csv(r"{}/eval.csv".format(eval_results_dir), index=False)

