import argparse
import os

import pandas as pd
import ir_measures
from ir_measures import *

from models.Citadel.citadel_retrieval import CitadelRetrieve
from utils.utils_memory import get_folder_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--dataset", type=str, default=r"antique")
    parser.add_argument("--data_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--queries_dir", type=str, default=r"./data/query")
    parser.add_argument("--ctx_embeddings_dir", type=str, default=r"./index/Citadel_new")
    parser.add_argument("--index_dir", type=str, default=r"./index/Citadel_new")

    parser.add_argument("--transformer_model_dir", type=str, default=r"./checkpoints/bert-base-uncased")

    parser.add_argument("--encode_batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tok_projection_dim", type=int, default=32)
    parser.add_argument("--cls_projection_dim", type=int, default=128)

    parser.add_argument("--check_point_path", type=str, default=r"./checkpoints/citadel.ckpt")
    parser.add_argument("--device", type=str, default=r"cpu")
    parser.add_argument("--encode_device", type=str, default=r"cpu")
    parser.add_argument("--content_topk", type=float, default=5)
    parser.add_argument("--retrieved_topk", type=float, default=30)

    parser.add_argument("--add_context_id", type=bool, default=False)
    parser.add_argument("--weight_threshold", type=float, default=0.5)
    parser.add_argument("--prune_weight", type=float, default=0.8)

    parser.add_argument("--cls_dim", type=float, default=128)
    parser.add_argument("--dim", type=float, default=32)
    parser.add_argument("--sub_vec_dim", type=float, default=4)
    parser.add_argument("--num_centroids", type=float, default=16)
    parser.add_argument("--iter", type=float, default=5)
    parser.add_argument("--threshold", type=int, default=1000)

    parser.add_argument("--query_json_dir", type=str, default=r'./data/query')
    parser.add_argument("--label_json_dir", type=str, default=r'./data/label')
    parser.add_argument("--results_save_to", type=str, default=r'./results_test2')
    parser.add_argument("--portion", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--measure", type=list, default=[nDCG@10, RR@10, Success@10])
    args = parser.parse_args()
    ######################################

    dataset_list = ["fiqa"]
    for dataset in dataset_list:
        args.dataset = dataset
        args.corpus_file = r"{}/{}.jsonl".format(args.data_dir, args.dataset)
        print(args.dataset)
        prune_weights_list = [0, 0.3, 0.6, 0.9]
        eval_list = []
        for prune_weight in prune_weights_list:
            args.prune_weight = str(prune_weight)
            print(args.prune_weight)
            expert_dir = r"{}/{}/{}/{}".format(args.index_dir, args.dataset,
                                               args.content_topk,
                                               r"expert", prune_weight)
            cls_path = os.path.join(args.index_dir, args.dataset, "cls.pkl")
            index_memory = get_folder_size(expert_dir)
            index_memory += os.path.getsize(cls_path)
            cr = CitadelRetrieve(args)
            path = cr.run()
            eval_results = cr.evaluate(path, index_memory)
            eval_list.append(eval_results)

        eval_df = pd.DataFrame(eval_list)
        eval_df.to_csv(r"{}/citadel/{}/eval_results/eval.csv".format(args.results_save_to, args.dataset), index=False)



