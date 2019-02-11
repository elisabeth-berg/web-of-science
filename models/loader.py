import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import casual_tokenize

def load_data():
    X = pd.read_table('data/X.txt', sep='\n', header=None)
    X.columns = ['text']
    X = X.apply(lambda x: x[0], axis=1)
    y = pd.read_table('data/YL1.txt', sep='\n', header=None)
    y.columns = ['label']
    y = y.apply(lambda x: x[0], axis=1)
    return X, y

def vectorize(X, min_df):
    X = X.apply(parse_doc)
    vectorizer = TfidfVectorizer(
        stop_words='english',
        min_df=min_df)
    X = vectorizer.fit_transform(X)
    return X, vectorizer.get_feature_names()

def parse_doc(doc):
    stemmer = SnowballStemmer("english")
    stems = []
    for word in casual_tokenize(doc):
        stem = stemmer.stem(word.lower())
        if stem.isalpha():
            stems.append(stem)
    return ' ' .join(stems)
