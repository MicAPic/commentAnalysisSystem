import copy
import re
import sys
from ast import literal_eval
from collections import Counter
from difflib import Differ
from itertools import chain

import gensim.corpora as corpora
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import spacy
from spacy.matcher import Matcher

import csv_rw

LANGUAGE = "en"
pipelines = {
    "en": "en_core_web_sm",
    "fr": "fr_dep_news_sm"
}


def generate_corpus_file(data_list, target_file_name="corpus.txt"):
    id2word = corpora.Dictionary(data_list)

    original_stdout = sys.stdout

    with open(f"corpora/{target_file_name}", 'w', encoding="utf-8") as f:
        sys.stdout = f
        for token in sorted(list(id2word.token2id.keys())):
            # print(token.replace('_', ' '))
            print(token)
        sys.stdout = original_stdout


def compare_dataframes(df1, corpus1: str, df2, corpus2: str):
    data_list1, data_list2 = get_tokens(df1), get_tokens(df2)
    keyword_list1, keyword_list2 = get_keywords(df1), get_keywords(df2)

    #                               removed_keywords, added_keywords
    # compare 2 corpora line-by-line
    added_keywords, removed_keywords = [], []
    with open(corpus1, 'r', encoding="utf-8") as file1, open(corpus2, 'r', encoding="utf-8") as file2:
        differ = Differ()
        # is this needed?
        for line in differ.compare(file1.readlines(), file2.readlines()):
            if line[0] == '+':
                added_keywords.append(line[2:-1])
            elif line[0] == '-':
                removed_keywords.append(line[2:-1])

    for keyword in copy.deepcopy(removed_keywords):
        if keyword in added_keywords:
            removed_keywords.remove(keyword)
            added_keywords.remove(keyword)

    data_list1_counted = Counter(list(chain.from_iterable(data_list1)))
    data_list2_counted = Counter(list(chain.from_iterable(data_list2)))
    # sort keywords by popularity
    removed_keywords = sorted(removed_keywords, key=data_list1_counted.get, reverse=True)
    added_keywords = sorted(added_keywords, key=data_list2_counted.get, reverse=True)
    #

    #                               trending_keywords, fading_away_keywords
    # get the difference between two dictionaries
    data_list2_counted.subtract(data_list1_counted)
    difference = dict(data_list2_counted.most_common())  # .most_common() keeps the sorting

    # remove topic keywords from the list
    topic_keywords = set(chain.from_iterable(keyword_list1)) | set(chain.from_iterable(keyword_list2))
    difference = {k: v for (k, v) in difference.items() if k not in topic_keywords}

    # use k-means to divide the dictionary of keyword popularity difference into 3 clusters
    y_pred = KMeans(n_clusters=3).fit_predict(np.asarray(list(difference.values())).reshape(-1, 1))
    plt.scatter(np.zeros(len(difference)), difference.values(), c=y_pred, marker=".", linewidths=0.1)
    plt.xticks([])
    plt.show()

    # assign variables to the clusters
    trending_keywords, fading_away_keywords = dict(), dict()
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i - 1]:
            trending_keywords = dict(list(difference.items())[0:i])
            break
    for i in range(len(y_pred) - 2, -1, -1):
        if y_pred[i] != y_pred[i + 1]:
            fading_away_keywords = dict(list(difference.items())[-1:i:-1])
            break
    #

    return removed_keywords, added_keywords, trending_keywords, fading_away_keywords, y_pred


def get_tokens(dataframe):
    token_list = list(map(lambda x: literal_eval(x), dataframe["Tokens"].values.tolist()))
    return token_list


def get_keywords(dataframe):
    keyword_list = list(map(lambda x: literal_eval(x), dataframe["Keywords"].values.tolist()))
    return keyword_list


def get_contribution(dataframe):
    contribution_list = dataframe["Contribution"].values.tolist()
    return contribution_list


def search(target_word, target_df, data_list, keyword_list, contribution_list):
    # target_word = target_word.replace(' ', '_')

    def input2expr(input_str: str):
        input_str = input_str.replace("(", "( ")
        input_str = input_str.replace(")", " )")
        input_str = ' '.join(["'" + word + "' in ֍" if word not in ('not', 'or', 'and', '(', ')')
                              else word
                              for word in input_str.split()])
        input_str = input_str.replace("( ", "(")
        input_str = input_str.replace(" )", ")")
        return input_str

    target_word = input2expr(target_word)
    target_df["Result"] = pd.Series
    for i in range(len(target_df)):
        if eval(target_word.replace('֍', str(data_list[i]))):
            target_df["Result"][i] = 1
        elif eval(target_word.replace('֍', str(keyword_list[i]))):
            target_df["Result"][i] = contribution_list[i]
        else:
            target_df["Result"][i] = 0


def rule_search(rule, target_df):
    nlp = spacy.load(pipelines.get(LANGUAGE, "en_core_web_sm"))
    matcher = Matcher(nlp.vocab)
    matcher.add("rule", rule)

    target_df["Result"] = pd.Series
    for i, text in enumerate(target_df["Text"].values):
        doc = nlp(text)
        target_df["Result"][i] = int(bool(matcher(doc)))


if __name__ == "__main__":
    results = "lda results (en)/table_no_2022-08-17_1512.tsv"
    df = csv_rw.read(results)
    # results1, results2 = "lda results (fr)/mini_table_no_2022-08-15_1307(dec2019).tsv", \
    #                      "lda results (fr)/mini_table_no_2022-08-15_1338(jan2020).tsv"
    # old_df, new_df = csv_rw.read(results1), csv_rw.read(results2)
    # old_data_list, new_data_list = get_tokens(old_df), get_tokens(new_df)
    # generate_corpus_file(old_data_list, "old_corpus.txt")
    # generate_corpus_file(new_data_list, "new_corpus.txt")
    # deprecated, added, trending, fading_away, y = compare_dataframes(old_df, "corpora/old_corpus.txt",
    #                                                                  new_df, "corpora/new_corpus.txt")
    data_list = get_tokens(df)
    keyword_list = get_keywords(df)
    contribution_list = get_contribution(df)

    keyword_df = pd.DataFrame()
    keyword_df["Text"] = df["Text"]
    # temp_df = pd.DataFrame(columns=sorted(list(id2word.token2id.keys())))
    # keyword_df = pd.concat([keyword_df, temp_df], axis=1)

    # word4search = input("Keyword: ")
    # search(word4search, keyword_df, data_list, keyword_list, contribution_list)

    rule4search = [[{"ORTH": "#"}, {"IS_ASCII": True, 'LIKE_NUM': False}]]
    rule_search(rule4search, keyword_df)
