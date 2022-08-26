import re
from datetime import datetime

import contractions
import gensim.corpora as corpora
import gensim.models
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer  # some IDEs may show a warning here, although this works fine
from gensim.models import CoherenceModel
from stop_words import get_stop_words
import pyLDAvis.gensim_models

import csv_rw
import entity_recognition
from usuk2ca import canadize

"""
Example: LANGUAGE = "en"
"""
LANGUAGE = ""
pipelines = {
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm"
}


def clean_up(texts):
    """
    Used in run_lda() below
    """
    for index, text in enumerate(texts):
        print(f'Clean-up progress: {round((index + 1) / len(texts), 4) * 100}%', end='\r')
        # remove english contractions
        text = re.sub(r"\\", "", text)
        text = re.sub(r"[’´`]", "'", text)
        if LANGUAGE == "en":
            text = contractions.fix(text)

        # remove e-mails and url links
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www.\S+', '', text)

        # recognize entities, exclude them from subsequent clean_up
        text, entity_list, hashtag_list = entity_recognition.exclude_tokens(text)

        # remove ordinal indicators
        if LANGUAGE == "en":
            text = re.sub(r"(?<=\d)(st|nd|rd|th)\b", '', text)
        # remove all non-alphabetic characters
        text = re.sub("[^a-zA-ZàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇα-ω]+", ' ', text)
        # text = re.sub("\u2063", '', text)
        # remove single letter words
        text = re.sub(r'\b[a-zA-ZàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]\b', '', text)
        # substitute multiple spaces for one
        text = re.sub(' {2,}', ' ', text)

        # partially fix elongated words
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # return the entities and matches
        for entity in entity_list:
            text = re.sub("εντιτυ", entity, text, count=1)
        for hashtag in hashtag_list:
            text = re.sub("χασταγ", hashtag, text, count=1)

        # remove leading and trailing whitespaces and underscores
        text = text.strip()
        text = text.strip("_")

        texts[index] = text.lower()


def process(texts):
    """
    Used in run_lda() below
    """
    # initialize the spacy pipeline
    nlp = spacy.load(pipelines.get(LANGUAGE, "en_core_web_sm"))
    nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)
    pos_tags = ['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'ADP']
    excluded_tokens = entity_recognition.ruler.ent_ids

    for index, text in enumerate(texts):
        print(f'Language processing progress: {round((index + 1) / len(texts), 4) * 100}%', end='\r')
        doc = nlp(text)

        if LANGUAGE == "en":
            # if this is english, convert the words to their canadian form
            # this is very important, especially for f###ing yoghurt, which has 4 correct spellings
            texts[index] = [canadize(str(token)) if (str(token) + "/l-excluded") in excluded_tokens
                            # dont lemmatize hashtags:
                            or entity_recognition.matcher(nlp(re.sub('#', "# ", doc[i:i+1].text, count=1)))
                            else canadize(token.lemma_) for i, token in enumerate(doc) if token.pos_ in pos_tags]
        else:
            texts[index] = [str(token) if (str(token) + "/l-excluded") in excluded_tokens
                            # again, dont lemmatize hashtags:
                            or entity_recognition.matcher(nlp(re.sub('#', "# ", doc[i:i+1].text, count=1)))
                            else token.lemma_ for i, token in enumerate(doc) if token.pos_ in pos_tags]


def remove_stopwords(texts):
    """
    Used in run_lda() below
    """
    stop_words = get_stop_words(LANGUAGE)
    stop_words.extend(["none", "as_well_as"])

    for index, entry in enumerate(texts):
        print(f'Stopword removal progress: {round((index + 1) / len(texts), 4) * 100}%', end='\r')
        new_entry = []
        for word in entry:
            # check if this is an ngram of stopwords (like "has_been")
            if '_' in word:
                ngram = word.split('_')
                for gram in ngram:
                    if gram not in stop_words:
                        new_entry.append(word)
                        break
                continue
            if (word not in stop_words) and (word != ''):
                new_entry.append(word)
        if new_entry and (len(' '.join(new_entry)) >= 20):  # it's not empty and at least 20 characters long
            texts[index] = new_entry
        else:
            texts[index] = []


def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Used in run_lda() below
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print(f'Coherence analysis progress: {round((num_topics + 1) / limit, 4) * 100}%', end='\r')
        model = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values


def run_lda(
        dataframe: pd.DataFrame,
        from_date: str,
        to_date: str,
        visualize=False
):
    """
    Run LDA on a given DataFrame with the specified time frame, then write the results at /lda results (LANGUAGE)/.

    :param dataframe: pandas DataFrame (should contain columns "insert_date", "text", "language")
    :param from_date: Start date & time in the format of YYYY-MM-DD HH:MM:SS (time is optional)
    :param to_date: End date & time in the format of YYYY-MM-DD HH:MM:SS (time is optional)
    :param visualize: If True, saves a pyLDAvis visualization of LDA at /LDAvis/ (*.html)
    """
    dataframe["insert_date"] = pd.to_datetime(dataframe["insert_date"])
    dataframe = dataframe.set_index("insert_date")

    df_sample = dataframe[from_date:to_date][["text", "language"]]
    # alternatively, you can translate the df and comment out the next line:
    df_sample = df_sample.loc[df_sample["language"] == LANGUAGE]
    #
    df_sample = df_sample.drop_duplicates()

    text_series = df_sample["text"]

    data_list = text_series.tolist()

    # clean up the text
    clean_up(data_list)
    # lemmatize individual words
    process(data_list)

    # group common collocations into bigrams and trigrams
    bigram = gensim.models.Phrases(data_list, connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)

    trigram = gensim.models.Phrases(bigram[data_list], connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    data_list = [bigram_mod[entry] for entry in data_list]
    data_list = [trigram_mod[bigram_mod[entry]] for entry in data_list]
    # remove stopwords
    remove_stopwords(data_list)

    # LDA: find best amount of topics (the numbers are somewhat arbitrary, you can tweak them)
    start, limit = [3, 25]
    # data_list = csv_rw.read("temp.tsv").values.tolist()
    # data_list = list(map(lambda x: ast.literal_eval(x[0][:]), data_list))
    id2word = corpora.Dictionary(data_list)
    corp = [id2word.doc2bow(text) for text in data_list]
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corp,
                                                            texts=data_list, start=start, limit=limit, step=1)
    x = range(start, limit, 1)
    plt.plot(x, coherence_values)
    # plot the mean for better data representation
    mean = sum(coherence_values) / float(len(coherence_values))
    print(f'Mean is {mean}')
    plt.plot(x, [mean] * len(x), '--')
    plt.xlabel("Num Topics")
    plt.xticks(range(min(x), max(x) + 1))
    plt.ylabel("Coherence score")
    plt.legend("Coherence values", loc='best')
    plt.grid(True)
    plt.show()

    optimal_topic_no = int(input("Enter the number of topics that gives the best coherence score: "))
    # optimal_topic_no = 14

    # LDA (optimal_topic_no)
    model = model_list[optimal_topic_no - start]

    # save a visualization of the results with the chosen optimal_topic_no, can be used to tweak it on subsequent runs
    if visualize:
        lda_visualization = pyLDAvis.gensim_models.prepare(model, corp, id2word)
        # alternatively, you can use pyLDAvis.show(...) or pyLDAvis.save_json(...)
        pyLDAvis.save_html(lda_visualization, f"LDAvis/vis_no_{datetime.now():%Y-%m-%d_%H%M}.html")

    topic_keywords = model.show_topics(optimal_topic_no)
    topic_keywords = list(map(lambda x: re.sub("""[-*.0123456789,+"]""", '', x[1]), topic_keywords))
    topic_keywords = list(map(lambda x: x.split("  "), topic_keywords))
    # use Wikidata Api to determine topics (UNUSED)
    # print("Superclasses found:")
    # topics = list(map(lambda x: determine_topic(x), topic_keywords))

    temp_df = pd.DataFrame()
    for i, row in enumerate(model[corp]):
        print(f'Dominant Topic Allocation Progress: {round((i + 1) / len(model[corp]), 4) * 100}%', end='\r')
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # get the Dominant topic and Perc Contribution for each comment
        for j, (topic_n, prop_topic) in enumerate(row):
            if j == 0:
                if data_list[i]:
                    temp_df = pd.concat(
                        [temp_df, pd.Series(
                            [topic_keywords[topic_n], data_list[i], topic_keywords[topic_n][0], round(prop_topic, 4)])],
                        axis=1)
                else:
                    # no topic
                    temp_df = pd.concat([temp_df, pd.Series(
                        [[], data_list[i], '', 0.0000])], axis=1)
            else:
                break

    df_sample = pd.concat([df_sample.reset_index(drop=True),
                           temp_df.transpose().reset_index(drop=True)], axis=1)
    df_sample.columns = ['Text', 'Language', 'Keywords', 'Tokens', 'Estimated Theme', 'Contribution']
    print(f"Final Dataframe:\n{df_sample}")
    csv_rw.write(df_sample, f"lda results ({LANGUAGE})/table_no_{datetime.now():%Y-%m-%d_%H%M}.tsv")


"""
Example:

if __name__ == '__main__':
    filename = "fixed_Full_data_grocery_20220704.csv"
    df = csv_rw.read(filename, 'λ')

    run_lda(df, '2020-01-01 00:00:00', '2020-01-02 00:00:00', visualize=True)
"""
