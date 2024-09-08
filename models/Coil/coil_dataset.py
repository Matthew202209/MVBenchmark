import json

from torch.utils.data import Dataset

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


    def load_queries(self):
        for idx, pa in enumerate(self.queries.values()):
            if pa == "":
                pa = "happy happy happy"
            encode = self.tok.encode(
                pa,
                add_special_tokens=False,
                max_length=self.p_max_len,
                truncation=True
            )
            self.nlp_dataset.append(encode)

    def __len__(self):
        return len(list(self.queries.keys()))

    def __getitem__(self, item):
        encoded_inputs = self.nlp_dataset[item]
        encoded_psg = self.tok.encode_plus(
            encoded_inputs,
            max_length=self.p_max_len,
            truncation='only_first',
            return_attention_mask=True,
        )
        return encoded_psg
