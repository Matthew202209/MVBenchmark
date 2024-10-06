import argparse
import os

from models.SLIM.slim_index import SlimIndex
cplus_include_path = r'/home/chunming/miniconda3/envs/faiss/include:/home/chunming/miniconda3/envs/faiss/x86_64-conda-linux-gnu'
os.environ['CPLUS_INCLUDE_PATH'] = cplus_include_path
from  pyserini import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default=r"./")
    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
    parser.add_argument("--data_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--ctx_embeddings_dir", type=str, default=r"./index")
    parser.add_argument("--transformer_model_dir", type=str, default=r"./checkpoints/bert-base-uncased")

    parser.add_argument("--vocab_file", type=str, default=r"./checkpoints/slim/vocab.txt")
    parser.add_argument("--index_dir", type=str, default=r"./index")
    parser.add_argument("--corpus_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--check_point_path", type=str, default=r"/home/chunming/data/chunming/projects/MVBenchmark/checkpoints/slim")

    parser.add_argument("--device", type=str, default=r"cuda:0")
    parser.add_argument("--weight_threshold", type=float, default=0.1)
    parser.add_argument("--encode_batch_size", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context_top_k", type=int, default=1)
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    for dataset in ["antique","arguana", "clinicaltrials", "dbpedia-entity",
                    "fiqa", "lotte_pooled_dev", "msmarco-passage","nfcorpus",
                    "quora","scidocs", "scifact"]:
        si = SlimIndex(args)
        si.setup()
        si.build_lucene_index()

