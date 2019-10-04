import itertools
import re
from string import punctuation
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
from sklearn.neighbors import LocalOutlierFactor
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.nan)

def clean_data(text):
    """
        remove emojies, numbers symbols,
        and apply  lemma to the text data
     """
    txt = str(text)
    # Emoji replacement
    txt = re.sub(r':\)', r' happy ', txt)
    txt = re.sub(r':D', r' happy ', txt)
    txt = re.sub(r':P', r' happy ', txt)
    txt = re.sub(r':\(', r' sad ', txt)

    # Remove punctuation from text
    txt = ''.join([c for c in txt if c not in punctuation])

    # Replace words like sooooooo with so
    txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))

    # Remove all symbols
    txt = re.sub(r'[^A-Za-z0-9\s]', r' ', txt)
    txt = re.sub(r'\n', r' ', txt)

    txt = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    txt = re.sub(r'\<a href', ' ', txt)
    txt = re.sub(r'&amp;', '', txt)
    txt = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', txt)
    txt = re.sub(r'<br />', ' ', txt)
    txt = re.sub(r'\'', ' ', txt)
    # remove numbers
    txt = re.sub(r'[0-9]', r' ', txt)

    stop_words = stopwords.words('english')
    txt = " ".join([w for w in txt.split() if w not in stop_words])
    return txt


def tfidf_vector(df, start_date, end_date):
    """
        create a vector matrix of the cleaned text data
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1))
    vectorizer.fit(df['review_text_lemma'])
    print(vectorizer.get_feature_names())
    # df_train, df_test = split_data(df, start_date, end_date)
    feature_vector = vectorizer.transform(df['review_text_lemma'])
    # feature_vector_test = vectorizer.transform(df_test['review_text_lemma'])
    col = ['feat_' + i for i in vectorizer.get_feature_names()]
    feature_vector = pd.DataFrame(feature_vector.todense(), columns=col)
    return feature_vector

def clean_vector(vector):
    """
        remove columns like text, date , rating from the feature vector created
    """
    cleaned_vector = vector
    for col in vector.columns:
        if not col.startswith('feat'):
            cleaned_vector = cleaned_vector.drop(col, 1)
    return cleaned_vector

def split_data(df, start_date, end_date):
    """
        split data into test and train according to the date
    """
    df_train = df.loc[(df['date'] <= start_date)]
    df_test = df.loc[(df['date'] > start_date) & (df['date'] < end_date)]
    return (df_train, df_test)

def accuracy(df):
    outlier = df.loc[(df['outlier'] == -1)]
    non_outlier = df.loc[(df['outlier']== 1)]
    print(len(outlier))
    print(len(non_outlier))

def hyperParameter():
    contamination = [0.1, 0.01, 0.5, 0.05, 0.09, 0.3, 0.06, 0.03]
    neighbours = [15, 20, 25, 30, 35, 40, 45, 50, 100, 150]
    for x in contamination:
        for y in neighbours:
            # print(x,y)
            return (x,y)
    return (0.01, 20)

def localOutlierFactor(df, start_date, end_date):
    print(df.head())
    feature_vector = tfidf_vector(df, start_date, end_date)
    # df_train, df_test = split_data(df, start_date, end_date)
    feature_vector = feature_vector.head(100)
    df = df.head(100)
    vectors = clean_vector(feature_vector)
    # print("vectors",vectors)
    model = LocalOutlierFactor(contamination=0.1, n_neighbors=20, algorithm = 'auto')
    outlier_pred = model.fit_predict(vectors)
    df['outlier'] = outlier_pred
    print(df.head())
    accuracy(df)
    return df


def main():
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('Mint.csv')
    print(df.head())
    df.sort_values(by='rundate', inplace=True, ascending=True)
    # amazonreviews = pd.read_csv('Reviews.csv')
    # print(amazonreviews.head())
    df.fillna(" ", inplace=True)
    # amazonreviews.fillna(" ", inplace=True)
    df['review_text_lemma'] = df['reviewtext'].map(lambda x: clean_data(x))
    # amazonreviews['review_text_lemma'] = amazonreviews['Text'].map(lambda x: clean_data(x))
    # amazonreviews.to_csv("amazon_reviews.csv", index=False)
    localOutlierFactor(df, "2018-02-01 00:00:00.0", "2018-05-01 00:00:00.0")


if __name__ == '__main__':
    main()

