# commentAnalysisSystem
This is a series of scripts dedicated to analyzing Internet comments about an undisclosed Canadian company.

### Instructions:

1. Run _lda.py_ on a .csv/.tsv/etc. file. The input table must be in the following format:

| insert_date         | text                               | language | ... |
|---------------------|------------------------------------|----------|-----|
| 2019-05-27 21:06:48 | keep up the good work              | en       | ... |
| 2019-05-27 21:06:48 | Vendez un Bar-B-Q déjà toute monté | fr       | ... |
| ...                 | ...                                | ...      | ... |
2. Afterwards, you can use 
   * _keyword_analysis.py_: corpus comparison based on the K-means algorithm, keyword search among comments using logical expressions or SpaCy rule-based matching;
   * _similarity_model.py_: word2vec model training on the obtained corpus and the search among keywords that is based on it.

### How to modify _patterns.jsonl_:
New entities for _entity_recognition.py_ script can be added here. This file uses SpaCy pattern keys. For more information, see [available token attributes](https://spacy.io/usage/rule-based-matching#adding-patterns-attributes) and [available labels](https://stackoverflow.com/questions/53383601/can-you-determine-list-of-labels-for-existing-entityrecognizer-ner).

Unicode symbols (such as _è_) must be written using their respective source code (in this case, _\u00e8_). Don't put any whitespaces, as they'll break the script.

Finally, if you wish to exclude a specific entity from SpaCy lemmatization (for example, you want _"food_basics"_ to always stay plural), add _"/l-excluded"_ to its id (see the file itself for some examples).

### How to modify _usuk2ca-dictionary.txt_:

This dictionary is based on a British-American one from [this](https://www.tysto.com/uk-us-spelling-list.html) site, and, though not comprehensive, it gets the job done.

New entries can be added in the `non-canadian-spelling'\t'canadian-spelling` format (see the file).