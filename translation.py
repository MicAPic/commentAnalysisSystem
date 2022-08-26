import requests
import translators as ts
import csv_rw

FILENAME = "Full_data_grocery_20220704.csv"

df = csv_rw.read("fixed_" + FILENAME)
df = df[["text", "language"]]

length = len(df["text"])
for i in range(length):
    print(f'Translation progress: {round((i + 1) / length, 5) * 100}% ({i}/{length})', end='\r')
    if df.iloc[i, 1] != "en":
        try:
            df.iloc[i, 0] = ts.bing(df.iloc[i, 0])
        except (requests.exceptions.HTTPError, KeyError):
            df.iloc[i, 0] = ts.google(df.iloc[i, 0])
        csv_rw.write(df, "translated_" + FILENAME)
print('\n')


