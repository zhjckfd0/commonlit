# -*- coding: utf-8 -*-
import re
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from nltk.corpus import stopwords

nltk.download('stopwords')
# specify GPU
device = torch.device("cuda")

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
        e=" ".join(e)
        
        excerpt_processed.append(e)
        
    return excerpt_processed 

train_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/train.csv")
test_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/test.csv")

train_df["excerpt_preprocessed"] = preprocess(train_df)
test_df["excerpt_preprocessed"] = preprocess(test_df)

X = train_df["excerpt_preprocessed"]
y = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

seq_len = [len(i.split()) for i in X_train]

pd.Series(seq_len).hist(bins = 30)

