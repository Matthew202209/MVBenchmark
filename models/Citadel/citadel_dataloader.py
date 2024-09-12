import json

import ujson
import ir_datasets
from tqdm import tqdm
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.corpus_file = config.corpus_file
        self.corpus = {}
        self.tokenizer = tokenizer
        self._load_corpus()

    def _load_corpus(self):
        num_lines = sum(1 for i in open(r"{}".format(self.corpus_file), 'rb'))
        with open(self.corpus_file, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    self.corpus[line.get("doc_id")] = line.get("text")

    def __len__(self):
        return len(list(self.corpus.keys()))

    def __getitem__(self, item):
        encoded_inputs = list(self.corpus.values())[item]
        encoded_psg = self.tokenizer.encode(
            encoded_inputs,
            add_special_tokens=False,
            max_length=self.config.max_seq_len,
            truncation=True
        )
        if len(encoded_psg) == 0:
            encoded_psg = [1]
        encoded_psg = self.tokenizer.encode_plus(
            encoded_psg,
            max_length=self.config.max_seq_len,
            truncation='only_first',
            return_attention_mask=True,
        )
        encoded_psg.data["corpus_ids"] = [item]

        return encoded_psg

class BenchmarkQueries(Dataset):
    def __init__(self, dataset, tokenizer, query_json_dir, p_max_len=128):
        self.dataset = dataset
        self.query_json_dir = query_json_dir
        self.nlp_dataset = None
        self.nlp_dataset = []
        self.tok = tokenizer
        self.p_max_len = p_max_len
        self._load_queries()

    def _load_queries(self):
        with open(r"{}/{}.json".format(self.query_json_dir, self.dataset), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)


class BenchmarkQueriesDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.dataset = self.config.dataset

        self.query_json_dir = config.query_json_dir
        self.queries = {}
        self.tokenizer = tokenizer
        self._load_queries()

    def _load_queries(self):
        with open(r"{}/{}.json".format(self.query_json_dir, self.dataset), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)

    def __len__(self):
        return len(list(self.queries.keys()))

    def __getitem__(self, item):
        encoded_inputs = list(self.queries.values())[item]
        encoded_psg = self.tokenizer.encode(
            encoded_inputs,
            add_special_tokens=False,
            max_length=self.config.max_seq_len,
            truncation=True
        )
        if len(encoded_psg) == 0:
            encoded_psg = [2476, 2476, 2476]
        encoded_psg = self.tokenizer.encode_plus(
            encoded_psg,
            max_length=self.config.max_seq_len,
            truncation='only_first',
            return_attention_mask=True,
        )

        return encoded_psg
