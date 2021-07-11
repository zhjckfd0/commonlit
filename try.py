# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 02:44:18 2021

@author: Yaroslav
"""
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
from wordcloud import WordCloud,STOPWORDS
from spacy import displacy
from nltk.tokenize import sent_tokenize, word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as mse

nltk.download('stopwords')
nltk.download('wordnet')

# os.environ["WANDB_SILENT"] = "true"

# wandb.login

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# =============================================
#functions start
# =============================================
def custom_palette(custom_colors):
    customPalette = sns.set_palette(sns.color_palette(custom_colors))
    sns.palplot(sns.color_palette(custom_colors),size=0.8)
    plt.tick_params(axis='both', labelsize=0, length = 0)

#====== Preprocessing function ======
def preprocess(data):
    excerpt_processed=[]
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
# ================================
#functions end
# ================================

palette = ["#7209B7","#3F88C5","#136F63","#F72585","#FFBA08"]
palette2 = sns.diverging_palette(120, 220, n=20)
custom_palette(palette)

custom_theme = Theme({
    "info" : "italic bold cyan",
    "warning": "italic bold magenta",
    "danger": "bold blue"
})

console = Console(theme=custom_theme)

# ========================================

train_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/train.csv")
test_df = pd.read_csv("F:/spyderProjects/CommonLit/CommonLit/test.csv")

msno.bar(train_df,color=palette[2], sort="ascending", figsize=(10,5), fontsize=12)
# plt.show()

excerpt1 = train_df['excerpt'].min()
console.print("Before preprocessing: ",style="info")
console.print(excerpt1,style='warning')

e = re.sub("[^a-zA-Z]", " ", excerpt1)
e = e.lower()
        
e = nltk.word_tokenize(e)
        
e = [word for word in e if not word in set(stopwords.words("english"))]
        
lemma = nltk.WordNetLemmatizer()
e = [lemma.lemmatize(word) for word in e]
e=" ".join(e)
console.print("After preprocessing: ",style="info")
console.print(e,style='warning')

# =============================================================================
# # =================================================
# # ARTIFACT
# # =================================================
# 
# train_df["excerpt_preprocessed"] = preprocess(train_df)
# test_df["excerpt_preprocessed"] = preprocess(test_df)
# 
# #====== Saving to csv files and creating artifacts ======
# train_df.to_csv("train_excerpt_preprocessed.csv")
# 
# run = wandb.init(project='commonlit', name='excerpt_preprocessed')
# 
# artifact = wandb.Artifact('train_excerpt_preprocessed', type='dataset')
# 
# #====== Add a file to the artifact's contents ======
# artifact.add_file("train_excerpt_preprocessed.csv")
# 
# #====== Save the artifact version to W&B and mark it as the output of this run ====== 
# run.log_artifact(artifact)
# 
# run.finish()
# 
# #====== Saving to csv files and creating artifacts ======
# test_df.to_csv("test_excerpt_preprocessed.csv")
# 
# run = wandb.init(project='commonlit', name='excerpt_preprocessed')
# 
# artifact = wandb.Artifact('test_excerpt_preprocessed', type='dataset')
# 
# #====== Add a file to the artifact's contents ======
# artifact.add_file("test_excerpt_preprocessed.csv")
# 
# #====== Save the artifact version to W&B and mark it as the output of this run ====== 
# run.log_artifact(artifact)
# 
# run.finish()
# 
# # ====================================================
# 
# run = wandb.init(project="commonlit")
# artifact = run.use_artifact(
#     "ruchi798/commonlit/train_excerpt_preprocessed:v0", type="dataset"
# )
# artifact_dir = artifact.download()
# run.finish()
# 
# path = os.path.join(artifact_dir, "train_excerpt_preprocessed.csv")
# train_df = pd.read_csv(path)
# train_df = train_df.drop(columns=["Unnamed: 0"])
# 
# run = wandb.init(project="commonlit")
# artifact = run.use_artifact(
#     "ruchi798/commonlit/test_excerpt_preprocessed:v0", type="dataset"
# )
# artifact_dir = artifact.download()
# run.finish()
# 
# path = os.path.join(artifact_dir, "test_excerpt_preprocessed.csv")
# test_df = pd.read_csv(path)
# test_df = test_df.drop(columns=["Unnamed: 0"])
# =============================================================================

#====== Function to plot wandb bar chart ======
# =============================================================================
# def plot_wb_bar(df,col1,col2,name,title): 
#     run = wandb.init(project='commonlit', job_type='image-visualization',name=name)
#     
#     dt = [[label, val] for (label, val) in zip(df[col1], df[col2])]
#     table = wandb.Table(data=dt, columns = [col1,col2])
#     wandb.log({name : wandb.plot.bar(table, col1,col2,title=title)})
# 
#     run.finish()
#     
# #====== Function to plot wandb histogram ======
# def plot_wb_hist(df,name,title):
#     run = wandb.init(project='commonlit', job_type='image-visualization',name=name)
# 
#     dt = [[x] for x in df[name]]
#     table = wandb.Table(data=dt, columns=[name])
#     wandb.log({name : wandb.plot.histogram(table, name, title=title)})
# 
#     run.finish()
# =============================================================================
    
fig, ax = plt.subplots(1,2,figsize=(20,10))
sns.kdeplot(train_df['target'], color=palette[0], shade=True,ax=ax[0])
sns.kdeplot(train_df['standard_error'], color=palette[1], shade=True,ax=ax[1])
ax[0].axvline(train_df['target'].mean(), color=palette[0],linestyle=':', linewidth=2)
ax[1].axvline(train_df['standard_error'].mean(), color=palette[1],linestyle=':', linewidth=2)
ax[0].set_title("Target Distribution",font="Serif")
ax[1].set_title("Standard Error Distribution",font="Serif")
ax[0].annotate('mean', xy=(-0.3* np.pi, 0.2), xytext=(1, 0.2), font='Serif',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
ax[1].annotate('mean', xy=(0.49, 6), xytext=(0.57, 6), font='Serif',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=-90"));
plt.show()

sns.jointplot(x=train_df['target'], y=train_df['standard_error'], kind='hex',height=10,edgecolor=palette[4])
plt.suptitle("Target vs Standard error ",font="Serif")
plt.subplots_adjust(top=0.95)
plt.show()    

# plot_wb_hist(train_df,"target","Target Distribution")
# plot_wb_hist(train_df,"standard_error","Standard Error Distribution")

plt.figure(figsize=(16, 8))
sns.countplot(y="license",data=train_df,palette="BrBG",linewidth=3)
plt.title("License Distribution",font="Serif")
plt.show()

def get_top_n_words(corpus, n=None):
    vec = CV().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_n_bigram(corpus, n=None):
    vec = CV(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def get_top_n_trigram(corpus, n=None):
    vec = CV(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plot_bt(x,w,p):
    common_words = x(train_df['excerpt_preprocessed'], 20)
    common_words_df = DataFrame (common_words,columns=['word','freq'])

    plt.figure(figsize=(16,8))
    sns.barplot(x='freq', y='word', data=common_words_df,facecolor=(0, 0, 0, 0),linewidth=3,edgecolor=sns.color_palette(p,20))
    plt.title("Top 20 "+ w,font='Serif')
    plt.xlabel("Frequency", fontsize=14)
    plt.yticks(fontsize=13)
    plt.xticks(rotation=45, fontsize=13)
    plt.ylabel("");
    return common_words_df

common_words = get_top_n_words(train_df['excerpt_preprocessed'], 20)
common_words_df1 = DataFrame(common_words,columns=['word','freq'])
plt.figure(figsize=(16, 8))
ax = sns.barplot(x='freq', y='word', data=common_words_df1,facecolor=(0, 0, 0, 0),linewidth=3,edgecolor=sns.color_palette("ch:start=3, rot=.1",20))

plt.title("Top 20 unigrams",font='Serif')
plt.xlabel("Frequency", fontsize=14)
plt.yticks(fontsize=13)
plt.xticks(rotation=45, fontsize=13)
plt.ylabel("");

common_words_df2 = plot_bt(get_top_n_bigram,"bigrams","ch:rot=-.5")
common_words_df3 = plot_bt(get_top_n_trigram,"trigrams","ch:start=-1, rot=-.6")