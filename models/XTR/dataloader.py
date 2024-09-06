import csv
import json
import os

# import pyterrier
import ujson
from tqdm import tqdm

import pyterrier as pt
if not pt.started():
    pt.init()


class BenchmarkDataLoader:
    def __init__(self, data_folder: str = None, corpus_file: str = "corpus.jsonl",
                 query_file: str = "qas.jsonl",
                 questions_file: str = "questions.tsv",):
        self.corpus = {}
        self.queries = {}
        self.questions = {}
        self.corpus_file = os.path.join(data_folder, corpus_file)
        self.query_file = os.path.join(data_folder, query_file)
        self.questions_file = os.path.join(data_folder, questions_file)

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def _load_corpus(self):

        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = ujson.loads(line)
                if '_id' in line.keys():
                    self.corpus[line.get('_id')] = line.get('text')
                elif "doc_id" in line.keys():
                    if line.get("text") == "":
                        print(line.get("doc_id"))
                    self.corpus[line.get("doc_id")] = line.get("text")

    def _load_queries(self):
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("qid")] = {"query": line.get("query") ,
                                                 "answer": line.get("answer_pids")}

    def _load_questions(self):
        reader = csv.reader(open(self.questions_file, encoding="utf-8"),
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for id, row in enumerate(reader):
            question_id, query = row[0], row[1]
            self.questions[question_id] = query

    def load(self):
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.questions_file, ext="tsv")

        if not len(self.corpus):
            print("Loading Corpus...")
            self._load_corpus()
            print("Loaded %d Documents.", len(self.corpus))
            print("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            print("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.questions_file):
            print("Loading question...")
            self._load_questions()

        return self.corpus, self.queries, self.questions

    def load_corpus(self):
        self.check(fIn=self.corpus_file, ext="jsonl")
        self._load_corpus()
        return self.corpus

    def load_queries(self):
        self.check(fIn=self.query_file, ext="jsonl")
        self._load_queries()
        return self.queries

    def load_questions(self):
        self.check(fIn=self.questions_file, ext="tsv")
        self._load_questions()
        return self.questions


class PTBenchmarkDataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.corpus = {}
        self.queries = {}

    def _load_queries(self, query_json_dir):
        with open(r"{}/{}.json".format(query_json_dir, self.dataset ), 'r', encoding="utf-8") as f:
            self.queries = json.load(f)

    def _load_corpus(self):
        data_doc = pt.get_dataset(self.dataset)
        self.corpus = {doc["docno"]: doc["text"] for doc in data_doc.get_corpus_iter()}

    def load_queries(self):
        self._load_queries()
        return self.queries

    def load_corpus(self):
        self._load_corpus()
