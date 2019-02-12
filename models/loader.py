
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import casual_tokenize

def load_data(label, proportion, min_df):
    X, y, y2 = importion()
    #Mechanical and Civil Engineering only
    X = X[(y==3) | (y==4)]
    y2 = y2[(y==3) | (y==4)]

    #Labels can be 13-22 inclusive
    y = label_picker(y2, label)

    #Remove rows to achieve the desired proportion
    n = int(np.floor((sum(y)/len(y)-0.05)*len(y)))
    idx = random.sample(set(y[y==1].index), n)
    y = y.drop(idx)
    X = X.drop(idx)

    X, vocab = vectorize(X, min_df)
    return X, y, vocab

def importion():
    X = pd.read_table('data/X.txt', sep='\n', header=None)
    X.columns = ['text']
    X = X.apply(lambda x: x[0], axis=1)
    y = pd.read_table('data/YL1.txt', sep='\n', header=None)
    y.columns = ['label']
    y = y.apply(lambda x: x[0], axis=1)
    y2 = pd.read_table('data/Y.txt', sep='\n', header=None)
    y2.columns = ['label']
    y2 = y2.apply(lambda x: x[0], axis=1)
    return X, y, y2

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

def label_picker(y, label):
    y = y.apply(lambda x: 0 if x != label else 1)
    return y
