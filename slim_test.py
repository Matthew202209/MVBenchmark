import argparse

from models.SLIM.slim_index import SlimIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_file", type=str, default=r"./")
    parser.add_argument("--dataset", type=str, default=r"nfcorpus")
    parser.add_argument("--data_dir", type=str, default=r"/home/chunming/projects/Mutivector/learn/plaidrepro/jsonl_data")
    parser.add_argument("--ctx_embeddings_dir", type=str, default=r"./index")
    parser.add_argument("--transformer_model_dir", type=str, default=r"./checkpoints/bert-base-uncased")

    parser.add_argument("--vocab_file", type=str, default=r"./checkpoints/slim/vocab.txt")
    parser.add_argument("--index_dir", type=str, default=r"./index")
    parser.add_argument("--corpus_dir", type=str, default=r"./data/corpus")
    parser.add_argument("--check_point_path", type=str, default=r"/home/chunming/Projects/Multivector/MVBenchmark/checkpoints/slim")

    parser.add_argument("--device", type=str, default=r"cuda:0")
    parser.add_argument("--weight_threshold", type=float, default=0.1)
    parser.add_argument("--encode_batch_size", type=int, default=25)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--context_top_k", type=int, default=1)

    args = parser.parse_args()

    si = SlimIndex(args)
    si.setup()
    si.run()

