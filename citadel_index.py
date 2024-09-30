import argparse

from models.Citadel.citadel_index import CitadelIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
    parser.add_argument("--metadata_dir", type=str, default=r"./data/metadata")
    parser.add_argument("--ctx_embeddings_dir", type=str, default=r"./index/Citadel")
    parser.add_argument("--transformer_model_dir", type=str, default=r"./checkpoints/bert-base-uncased")

    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tok_projection_dim", type=int, default=32)
    parser.add_argument("--cls_projection_dim", type=int, default=128)

    parser.add_argument("--check_point_path", type=str, default=r"./checkpoints/citadel.ckpt")
    parser.add_argument("--device", type=str, default=r"cuda:0")

    parser.add_argument("--add_context_id", type=bool, default=False)
    parser.add_argument("--weight_threshold", type=float, default=0)
    parser.add_argument("--content_topk", type=float, default=1)
    parser.add_argument("--prune_weights_list", type=list, default=[0, 0.3, 0.6, 0.9, 1.2])
    parser.add_argument("--cls_dim", type=float, default=128)
    parser.add_argument("--dim", type=float, default=32)
    parser.add_argument("--sub_vec_dim", type=float, default=4)
    parser.add_argument("--num_centroids", type=float, default=16)
    parser.add_argument("--iter", type=float, default=5)
    parser.add_argument("--threshold", type=int, default=1000)
    parser.add_argument("--num", type=int, default=-1)

    parser.add_argument("--is_parallel", type=bool, default=True)
    parser.add_argument("--num_threads", type=int, default=20)

    args = parser.parse_args()
    dataset_list = ["antique","arguana", "clinicaltrials", "dbpedia-entity",
                    "fiqa", "lotte_pooled_dev", "msmarco-passage","nfcorpus",
                    "quora","scidocs", "scifact", "car"]

    for dataset in ["lotte_pooled_dev"]:
        for content_topk in [5]:
            print("new parallel")
            print(r"batch_size :{}".format(args.encode_batch_size))
            args.dataset = dataset
            args.content_topk = content_topk
            citadel = CitadelIndex(args)
            citadel.parallel_run()
