import copy
from datetime import datetime

from gensim.models import Word2Vec

import csv_rw
from keyword_analysis import get_tokens


def train_model(
        table_path: str
):
    """
    Train a Word2Vec model based on a dataframe of comments located at table_path (should be already processed by
    lda.py).

    The model is saved at /models/word2vec/

    :param table_path: Path to the file
    """
    df = csv_rw.read(table_path)
    sentence_list = get_tokens(df)
    new_model = Word2Vec(sentences=sentence_list, vector_size=300, min_count=1, workers=4, sg=1)
    new_model.save(f"models/word2vec/no_{datetime.now():%Y-%m-%d_%H%M}.model")


def find_exact_matches(
        w2v_model: Word2Vec,
        target_word: str,
        n=7
) -> list[str]:
    """
    Search for closest matches that contain the given target_word among the keywords of a pre-trained W2V model

    :param w2v_model: Pre-trained W2V model
    :param target_word: Word (or a part of it) for search
    :param n: No. of matches to return
    :return: List of n best matches
    """

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
        return sorted(res, key=res.__getitem__, reverse=True)[0:n]
    else:
        return sorted(res, key=res.__getitem__, reverse=True)[1:n]


def find_most_similar(
        w2v_model: Word2Vec,
        target_word: str,
        n=7
) -> list[str]:
    """
    Search for matches that are similar in meaning to the given target_word among the keywords of a pre-trained
    W2V model (see https://en.wikipedia.org/wiki/Word2vec)

    :param w2v_model: Pre-trained W2V model
    :param target_word: Word for search
    :param n: No. of matches to return
    :return: List of n best matches
    """
    if target_word not in w2v_model.wv.index_to_key:
        w2v_model = copy.deepcopy(w2v_model)
        w2v_model.build_vocab([[target_word]], update=True)

    target_word = target_word.replace(' ', '_')
    corpus = w2v_model.wv.key_to_index.keys()
    similarity_values = w2v_model.wv.most_similar(target_word, topn=None)
    res = dict(zip(corpus, similarity_values))
    # print first n most similar keywords to the target_word
    return sorted(res, key=res.__getitem__, reverse=True)[1:n]


"""
Example:

if __name__ == "__main__":
    filename = "lda results (en)/table_no_2022-08-02_1058.tsv"
    train_model(filename)
    # or
    model = Word2Vec.load("models/word2vec/no_2022-08-09_1427.model")

    while True:
        target = input("Search: ")
        # apple
        print("Exact matches:", end=' ')
        print(find_exact_matches(model, target, 5))
        print("Most similar", end=' ')
        print(find_most_similar(model, target, 5))
        print()
        # Exact matches: ['apple', 'gala_apple', 'empire_apple', 'apple_orange', 'apple_pie']
        # Most similar['tomato', 'pear', 'celery', 'sweet_potato', 'zucchini']
        #
"""