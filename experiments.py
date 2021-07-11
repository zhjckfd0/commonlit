# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import re
import nltk
import textstat
import time
import wandb
import rich
import spacy

from pandas import DataFrame
from matplotlib.lines import Line2D
from rich.console import Console
from rich import print
from rich.theme import Theme
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from spacy import displacy
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse

nltk.download('stopwords')
nltk.download('wordnet')

custom_theme = Theme({
    "info": "italic bold cyan",
    "warning": "italic bold magenta",
    "danger": "bold blue"
})

console = Console(theme=custom_theme)

train_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/train.csv")
test_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/test.csv")


# ====== Preprocessing function ======
def preprocess(data):
    excerpt_processed = []
    for e in data['excerpt']:
        # find alphabets
        e = re.sub("[^a-zA-Z]", " ", e)

        # convert to lower case
        e = e.lower()

        # tokenize words
        e = nltk.word_tokenize(e)

        # remove stopwords
        e = [word for word in e if not word in set(stopwords.words("english"))]

        # lemmatization
        lemma = nltk.WordNetLemmatizer()
        e = [lemma.lemmatize(word) for word in e]
        e = " ".join(e)

        excerpt_processed.append(e)

    return excerpt_processed


def training(model, X_train, y_train, X_test, y_test):
    # model = make_pipeline(
    #     TfidfVectorizer(binary=True, ngram_range=(1,1)),
    #     model,
    # )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MSE = mse(y_test, y_pred)
    print("MSE: ", MSE)


train_df["excerpt_preprocessed"] = preprocess(train_df)
test_df["excerpt_preprocessed"] = preprocess(test_df)

# ridge = Ridge(fit_intercept = True, normalize = False)

my_model = RandomForestClassifier()

X = train_df["excerpt_preprocessed"]
y = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

training(model=my_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

# model = make_pipeline(
#         TfidfVectorizer(binary=True),
#         ridge,
#     )

my_model.fit(X, y)
test_pred = my_model.predict(test_df["excerpt_preprocessed"])

predictions = pd.DataFrame()
predictions['id'] = test_df['id']
predictions['target'] = test_pred
predictions.to_csv("F:\pycharm_projects\CommonLit\submission.csv", index=False)
predictions
