import ujson
from tqdm import tqdm


def create_new_2_old_list(corpus_file):
    new_2_old = []
    num_lines = sum(1 for i in open(corpus_file, 'rb'))
    with open(corpus_file, encoding='utf8') as fIn:
        for line in tqdm(fIn, total=num_lines):
            line = ujson.loads(line)
            new_2_old.append(line.get('doc_id'))

    return new_2_old