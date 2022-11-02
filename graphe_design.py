import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertTokenizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample, seed
import re
import os
import argparse
import pickle

import random

from pyencoders import DISTILBERT_PATH, BrownianBridgeLoss, BrownianEncoder, MLP

import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

data_dir = "datasets"

features = pd.read_csv("C:\\Users\\EnzoT\\Documents\\results\\songs\\features\\features.csv", sep=";")
songs = pd.read_csv(os.path.join(data_dir, "songs.csv"), encoding='utf-8', sep=";")

with open(os.path.join("model","test_data_z_100.pkl"), 'rb') as f:
    test_data = pickle.load(f)

for data in test_data:
    if data['is_author']:
        author = data["sentence"].replace("_AUTHOR", "")
    else:
        data["z"] = data["z"].squeeze()

    data["author"] = author


df = pd.DataFrame(test_data)

df_z = pd.DataFrame(df["z"].tolist()).add_prefix('z')

df = df.drop('z', axis=1)


method = 'TSNE'
if method=='PCA':
    pca=PCA(n_components=2)
    X=pca.fit_transform(df_z)
elif method=='TSNE':
    tsne=TSNE(n_components=2, n_jobs=4, init='pca', perplexity=30, early_exaggeration=12, random_state=13, verbose=1)
    X=tsne.fit_transform(df_z)
elif method=='Isomap':
    isomap=Isomap(n_neighbors=50, n_components=2)
    X=isomap.fit_transform(df_z)

df['X'] = X[:,0]
df['Y'] = X[:,1]

df_fig = df[df['author'].isin(["Rihanna"])].drop_duplicates('sentence')

fig = px.line(df, x="X", y="Y", color="author", markers=True, 
              text="sentence_id",
              hover_data={'author':False, 
                        'sentence':True,
                        'X':False,
                        'Y':False,
                        'is_author':False}, title="Songs representation using TSNE")
fig.show()


df_fig = df[df['sentence'].str.contains('_AUTHOR')].drop_duplicates('sentence')

fig = px.scatter(df_fig, x='X', y='Y', color='author', text='author', title="Author representations projected with %s" % method)
fig.update_traces(textposition="bottom left")
fig.show()
for trace in fig['data']: 
    if('author' in trace['name']): trace['showlegend'] = False

fig.show()

author_embeddings = df[df['sentence'].str.contains('_AUTHOR')].drop_duplicates('sentence').sort_values("author")
author_embeddings = pd.DataFrame(author_embeddings["z"].tolist()).add_prefix('z')
author_embeddings = np.array(author_embeddings)

from regressor import style_embedding_evaluation

features=features.drop(['id'], axis=1).groupby('author').mean().reset_index().sort_values("author")

style_embedding_evaluation(author_embeddings, features)