import json
import os

import ir_datasets
import pandas as pd

QUERY_DIR = {
    "nfcorpus" : "beir/nfcorpus/test",
    "fiqa" : "beir/fiqa/test",
    "quora": "beir/quora/dev",
    "scifact": "beir/scifact/test",
    "scidocs": "beir/scidocs",
    "antique": "antique/train/split200-valid",
    "arguana": 'beir/arguana',
    "clinicaltrials": 'clinicaltrials/2021/trec-ct-2021',
    "lotte": "lotte/pooled/dev/forum",
    "msmarco-passage":"msmarco-passage/trec-dl-2019",
    "dbpedia-entity":"beir/dbpedia-entity/test",
    "car":"car/v1.5/test200"
}



def create_and_save_json_query(dataset, save_path):
    save_path = r"{}/{}.json".format(save_path, dataset)
    dir = QUERY_DIR[dataset]
    dataset = ir_datasets.load(dir)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    with open(save_path, 'w') as f:
        json.dump(queries, f, indent=4)



def create_and_save_json_label(dataset_name, save_path):
    save_path = r"{}/{}.csv".format(save_path, dataset_name)
    dir = QUERY_DIR[dataset_name]
    dataset = ir_datasets.load(dir)
    qrels_list = []
    for qrel in dataset.qrels:
        qrels_dict = {"query_id": str(qrel.query_id),
                      "doc_id": str(qrel.doc_id),
                      "relevance": qrel.relevance}

        qrels_list.append(qrels_dict)
    qrels = pd.DataFrame(qrels_list)
    qrels.to_csv(save_path, index=False)

def create_json_corpus():
    pass
def create_json():
    pass

if __name__ == '__main__':
    root = r"../data"
    dataset_list = ["lotte", "msmarco-passage", "dbpedia-entity", "car"]
    for dataset in dataset_list:
        label_save_path = r"/home/chunming/data/chunming/projects/MVBenchmark/data/label"
        query_save_path = r"/home/chunming/data/chunming/projects/MVBenchmark/data/query"

        if not os.path.exists(label_save_path):
            os.mkdir(label_save_path)
        if not os.path.exists(query_save_path):
            os.mkdir(query_save_path)

        create_and_save_json_query(dataset, query_save_path)
        create_and_save_json_label(dataset, label_save_path)
