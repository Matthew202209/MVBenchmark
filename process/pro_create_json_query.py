import ir_datasets
import pandas as pd

QUERY_DIR = {
    "nfcorpus" : "beir/nfcorpus/test"
}



def create_and_save_json_query(dataset):
    dir = QUERY_DIR[dataset]
    dataset = ir_datasets.load(dir)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    return queries


def create_and_save_json_label(dataset):
    dir = QUERY_DIR[dataset]
    dataset = ir_datasets.load(dir)
    qrels_list = []
    for qrel in dataset.qrels:
        qrels_dict = {"query_id": qrel.query_id,
                      "doc_id": qrel.doc_id,
                      "relevance": qrel.relevance}

        qrels_list.append(qrels_dict)
    qrels = pd.DataFrame(qrels_list)
    return qrels

def create_json_corpus():
    pass
def create_json():
    pass