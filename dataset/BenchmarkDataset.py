import ujson
from torch.utils.data import Dataset
from tqdm import tqdm


class BenchmarkDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.corpus = {}
        self.tokenizer = tokenizer

    def _load_corpus(self):
        corpus_file = r"{}/{}.jsonl".format(self.config.corpus_dir, self.config.dataset)
        num_lines = sum(1 for i in open(corpus_file, 'rb'))
        with open(corpus_file, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    if line.get("text") == "":
                        print(1)
                    self.corpus[line.get("doc_id")] = line.get("text")

    def load_corpus(self):
        self._load_corpus()

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


class BenchmarkQueriesDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.queries_path = config.queries_path
        self.queries = {}
        self.tokenizer = tokenizer
        self._load_queries()

    def _load_queries(self):
        dataset = ir_datasets.load(self.queries_path)
        self.queries = {query.query_id: query.text for query in dataset.queries_iter()}

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

        encoded_psg = self.tokenizer.encode_plus(
            encoded_psg,
            max_length=self.config.max_seq_len,
            truncation='only_first',
            return_attention_mask=True,
        )

        return encoded_psg