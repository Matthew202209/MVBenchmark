import argparse
import gzip
import os

import faiss
import ir_measures
import pandas as pd
from ir_measures import *

from models.Coil.coil_retrieval import CoilRetriever
from process.pro_data import create_new_2_old_list
from utils.utils_memory import memory_usage, get_folder_size


def set_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="./checkpoints/coil_checkpoint")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased")
    parser.add_argument("--token_dim", type=str, default=32)
    parser.add_argument("--cls_dim", type=str, default=768)
    parser.add_argument("--token_rep_relu", type=bool, default=False)
    parser.add_argument("--token_norm_after", type=bool, default=False)
    parser.add_argument("--cls_norm_after", type=bool, default=False)
    parser.add_argument("--x_device_negatives", type=bool, default=False)
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--no_sep", type=bool, default=False)
    parser.add_argument("--no_cls", type=bool, default=False)
    parser.add_argument("--cls_only", type=bool, default=False)
    args = parser.parse_args()
    return args


def set_data_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_group_size", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="nfcorpus")
    parser.add_argument("--query_json_dir", type=str, default='./data/query')
    parser.add_argument("--encoded_save_path", type=str, default=r"./queries_encode")
    parser.add_argument("--doc_index_save_path", type=str, default=r"./index/Coil")
    parser.add_argument("--dataset_path", type=str, default=r"/home/chunming/projects/Mutivector/learn/plaidrepro/jsonl_data")
    parser.add_argument("--document", type=bool, default=False)
    parser.add_argument("--p_max_len", type=int, default=32)

    args = parser.parse_args()
    return args


def training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", type=str, default=False)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=128)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    args = parser.parse_args()
    return args

def evaluation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encode_query', type=str,
                        default=r"./queries_encode")
    parser.add_argument("--json_dir_root", type=str, default='./data')
    parser.add_argument('--measure', type=list,
                        default=[nDCG@10, RR @ 10, Success @ 10])
    parser.add_argument('--top', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument('--results_save_to', type=str,
                        default=r'./results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model_args = set_model_args()
    data_args = set_data_args()
    training_args = training_args()
    eval_args = evaluation_args()
    dataset_list = ["antique", "arguana", "clinicaltrials", "fiqa", "nfcorpus", "quora", "scidocs", "scifact"]
    for dataset in dataset_list:

        data_args.dataset = dataset

        save_dir = r"{}/coil/{}".format(eval_args.results_save_to, data_args.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        perf_path = r"{}/{}".format(save_dir, "perf_results")
        rank_path = r"{}/{}".format(save_dir, "rank_results")
        eval_results_dir = r"{}/{}".format(save_dir, "eval_results")
        index_load_path = r"{}/{}".format(data_args.doc_index_save_path, data_args.dataset)
        index_memory = get_folder_size(index_load_path)


        if not os.path.exists(perf_path):
            os.makedirs(perf_path)
        if not os.path.exists(rank_path):
            os.makedirs(rank_path)
        if not os.path.exists(eval_results_dir):
            os.makedirs(eval_results_dir)

        label_json_dir = r"{}/label".format(eval_args.json_dir_root)
        corpus_file = r"{}/corpus/{}.jsonl".format(eval_args.json_dir_root, data_args.dataset)
        qrels = pd.read_csv(r"{}/{}.csv".format(label_json_dir, data_args.dataset))
        qrels["query_id"] = qrels["query_id"].astype(str)
        qrels["doc_id"] = qrels["doc_id"].astype(str)
        new2old = create_new_2_old_list(corpus_file)


        coil_r = CoilRetriever(model_args, data_args, training_args, eval_args.device)
        coil_r.set_model()
        coil_r.set_data_loader()
        coil_r.load_index()


        scores, indices, perf_df = coil_r.retrieve(topk=eval_args.top)
        perf_df.to_csv(r"{}/coil_perf.csv".format(perf_path), index=False)
        rh = faiss.ResultHeap(scores.shape[0], eval_args.depth)
        rh.add_result(-scores.numpy(), indices.numpy())
        rh.finalize()
        corpus_scores, corpus_indices = (-rh.D).tolist(), rh.I.tolist()
        qid_list = list(coil_r.query_dataset.queries.keys())
        path = r"{}/coil.run.gz".format(rank_path)

        with gzip.open(path, 'wt') as fout:
            for i in range(len(corpus_scores)):
                q_id = qid_list[i]
                scores = corpus_scores[i]
                indices = corpus_indices[i]
                for j in range(len(scores)):
                    fout.write(f'{q_id} 0 {indices[j]} {j} {scores[j]} run\n')

        rank_results_pd = pd.DataFrame(list(ir_measures.read_trec_run(path)))

        for i, r in rank_results_pd.iterrows():
            rank_results_pd.at[i, "doc_id"] = new2old[int(r["doc_id"])]
        eval_results = ir_measures.calc_aggregate(eval_args.measure, qrels, rank_results_pd)
        eval_results["parameter"] = -1
        eval_results["index_memory"] = index_memory
        eval_results["index_dlen"] = len(new2old)
        eval_df = pd.DataFrame([eval_results])
        eval_df.to_csv(r"{}/eval.csv".format(eval_results_dir), index=False)