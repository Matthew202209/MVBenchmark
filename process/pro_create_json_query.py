import json
import os

import ir_datasets
import pandas as pd

QUERY_DIR = {
    "nfcorpus" : "beir/nfcorpus/test",
    "fiqa" : "beir/fiqa/test",
}



def create_and_save_json_query(dataset, save_path):
    dir = QUERY_DIR[dataset]
    dataset = ir_datasets.load(dir)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    with open(save_path, 'w') as f:
        json.dump(queries, f, indent=4)



def create_and_save_json_label(dataset, save_path):
    dir = QUERY_DIR[dataset]
    dataset = ir_datasets.load(dir)
    qrels_list = []
    for qrel in dataset.qrels:
        qrels_dict = {"query_id": qrel.query_id,
                      "doc_id": qrel.doc_id,
                      "relevance": qrel.relevance}

        qrels_list.append(qrels_dict)
    qrels = pd.DataFrame(qrels_list)
    qrels.to_csv(r"{}/{}.csv".format(save_path, dataset), index=False)

def create_json_corpus():
    pass
def create_json():
    pass

if __name__ == '__main__':
    root = r"../data"
    dataset = 'nfcorpus'
    label_save_path = r"../data/label".format(root)
    query_save_path = r"../data/query".format(root)

    if not os.path.exists(label_save_path):
        os.mkdir(label_save_path)
    if not os.path.exists(query_save_path):
        os.mkdir(query_save_path)

    create_and_save_json_query(dataset, query_save_path)
    create_and_save_json_label(dataset, label_save_path)
