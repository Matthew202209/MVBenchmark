from torch.utils.data import Dataset
from tqdm import tqdm
import ujson

class SlimDataset(Dataset):
    def __init__(self, config, transform=None, sep_token=" [SEP] "):
        self.config = config
        self.sep_token = sep_token
        self.text_transform = transform
        self.corpus = {}
        self._load_corpus()
    def __len__(self):
        return len(list(self.corpus))

    def __getitem__(self, item):

        corpus_text = list(self.corpus.values())[item]
        corpus_text = " ".join([self.sep_token, corpus_text])
        ctx_tensors = self.text_transform(corpus_text)
        ctx_tensors.data["corpus_ids"] = [str(item)]
        return ctx_tensors

    def _transform(self, text):
        result = self.text_transform(text)
        return result

    def _load_corpus(self):
        corpus_file = r"{}/{}.jsonl".format(self.config.data_dir, self.config.dataset)
        num_lines = sum(1 for i in open(corpus_file, 'rb'))
        with open(corpus_file, encoding='utf-8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    self.corpus[line.get("doc_id")] = line.get("text")
    
    def get_lucene_data(self):
        lucene_data = []
        for idx, contents in enumerate(self.corpus.values()):
            if contents == "":
                lucene_data.append({"id": str(idx), "contents":"[SEP]"})
            else:
                lucene_data.append({"id":str(idx), "contents": contents})
        return lucene_data

    
                