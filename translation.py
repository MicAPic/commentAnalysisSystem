import pandas as pd
import requests
import translators as ts
import csv_rw


def translate_df(
        file_name: str
):
    """
    Translates a given table of comments into English

    :param file_name: Name of a table to read (should contain columns "text", "language")
    """

    df = csv_rw.read(file_name)
    df = df[["text", "language"]]
    length = len(df["text"])

    for i in range(length):
        print(f'Translation progress: {round((i + 1) / length, 5) * 100}% ({i}/{length})', end='\r')
        if df.iloc[i, 1] != "en":
            try:
                df.iloc[i, 0] = ts.bing(df.iloc[i, 0])
            except (requests.exceptions.HTTPError, KeyError):
                df.iloc[i, 0] = ts.google(df.iloc[i, 0])
            csv_rw.write(df, "translated_" + file_name)
    print('\n')


"""
Example:

if __name__ == "__main__":
    filename = "fixed_Full_data_grocery_20220704.csv"
    translate_df(filename)
"""



