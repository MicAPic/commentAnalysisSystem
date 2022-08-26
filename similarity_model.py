import copy
from datetime import datetime

from gensim.models import Word2Vec

import csv_rw
from keyword_analysis import get_tokens

FILENAME = "lda results (en)/table_no_2022-08-02_1058.tsv"


def train_model(table_name: str):
    df = csv_rw.read(table_name)
    sentence_list = get_tokens(df)
    new_model = Word2Vec(sentences=sentence_list, vector_size=300, min_count=1, workers=4, sg=1)
    new_model.save(f"models/word2vec/no_{datetime.now():%Y-%m-%d_%H%M}.model")


def find_exact_matches(w2v_model, target_word, n=7):
    is_in_vocab = True

    if target_word not in w2v_model.wv.index_to_key:
        w2v_model = copy.deepcopy(w2v_model)
        w2v_model.build_vocab([[target_word]], update=True)
        is_in_vocab = False

    target_word = target_word.replace(' ', '_')
    corpus = w2v_model.wv.key_to_index.keys()
    similarity_values = w2v_model.wv.most_similar(target_word, topn=None)
    res = dict(zip(corpus, similarity_values))
    res = {k : v for k, v in res.items() if target_word in k}
    # print first n most similar keywords to the target_word
    if is_in_vocab:
        print(sorted(res, key=res.__getitem__, reverse=True)[0:n])
    else:
        print(sorted(res, key=res.__getitem__, reverse=True)[1:n])


def find_most_similar(w2v_model, target_word, n=7):

    if target_word not in w2v_model.wv.index_to_key:
        w2v_model = copy.deepcopy(w2v_model)
        w2v_model.build_vocab([[target_word]], update=True)

    target_word = target_word.replace(' ', '_')
    corpus = w2v_model.wv.key_to_index.keys()
    similarity_values = w2v_model.wv.most_similar(target_word, topn=None)
    res = dict(zip(corpus, similarity_values))
    # print first n most similar keywords to the target_word
    print(sorted(res, key=res.__getitem__, reverse=True)[1:n])


if __name__ == "__main__":
    # train_model(FILENAME)
    model = Word2Vec.load("models/word2vec/no_2022-08-09_1427.model")

    while True:
        target = input("Search: ")
        print("Exact matches:", end=' ')
        find_exact_matches(model, target, 10)
        print("Most similar", end=' ')
        find_most_similar(model, target, 10)
        print()
