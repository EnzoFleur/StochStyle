import pandas as pd
from sklearn.model_selection import train_test_split
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

# from regressor import style_embedding_evaluation
# from extractor import features_array_from_string

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

BATCH_SIZE = 32
EPOCHS = 100
NEGPAIRS = 10
LEARNING_RATE = 1e-3

data_dir = "datasets"

# songs = pd.read_csv(os.path.join(data_dir, "songs.csv"), encoding='utf-8', sep=";")

class SongTripletDataset(Dataset):

    def __init__(self, train, seed, sentence_size=1):
        super(SongTripletDataset, self).__init__()

        self.train = train
        self.seed = seed
        self.tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
        self.max_length = 512
        self.sentence_size = sentence_size

        self.data = pd.read_csv(os.path.join(data_dir, "songs.csv"), encoding='utf-8', sep=";").sort_values(by=["author", "title"])

        if self.train:
            self.data, _ = train_test_split(
                                        self.data,
                                        test_size=0.2,
                                        stratify=self.data[['author']],
                                        random_state=self.seed
                                        )
        else:
            _, self.data = train_test_split(
                            self.data,
                            test_size=0.2,
                            stratify=self.data[['author']],
                            random_state=self.seed
                            )

    def _process_data(self):

        self.authors = list(self.data.author)
        self.docs = list(self.data.lyrics)

        self.processed_data = []

        doc_id = 0

        for author, doc in zip(self.authors, self.docs):
            
            temp = doc.split('\n')
            sentences = [ '\n'.join(temp[i:i+self.sentence_size]) for i in range(0, len(temp),self.sentence_size)]

            # Add the author as starting point of the document
            self.processed_data.append({
                "sentence": "%s_AUTHOR" % author,
                "sentence_id": 0,
                "doc_id": doc_id,
                "total_doc_sentences": len(sentences),
                "is_author": True
            })

            for sentence_id, sentence in enumerate(sentences, start=1):
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "doc_id": doc_id,
                    "total_doc_sentences": len(sentences),
                    "is_author": False
                }
                self.processed_data.append(sentence_info)

            # Add the author as ending point of the document
            self.processed_data.append({
                "sentence": "%s_AUTHOR" % author,
                "sentence_id": sentence_id+1,
                "doc_id": doc_id,
                "total_doc_sentences": len(sentences),
                "is_author": True
            })

            doc_id += 1

        print("Length of dataset: %d" % len(self.processed_data))
        print("Number of documents: %d" % (doc_id))

    def get_tokenized(self, sentence):
        tokenized = self.tokenizer(sentence, truncation=True, max_length=self.max_length)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        input_ids += [self.tokenizer.eos_token_id] * (self.max_length - len(input_ids))
        attention_mask += [0] * (self.max_length - len(attention_mask))

        return input_ids, attention_mask

    def tokenize_caption(self, caption, device):

        output = self.tokenizer(caption, padding=True, return_tensors='pt')

        input_ids = output['input_ids']
        attention_mask = output['attention_mask']

        return input_ids.to(device), attention_mask.to(device)

    def __getitem__(self, index):
        item = self.processed_data[index]
        sentence_num = item['sentence_id']

        if sentence_num == 0:
            index +=2
        if sentence_num == 1:
            index+=1

        item = self.processed_data[index]
        sentence_num = item['sentence_id']

        T = sentence_num
        # t is a random point in between
        nums = list(range(T))
        t1 = random.choice(nums)
        nums.remove(t1)
        t2 = random.choice(nums)
        if t2 < t1:
            t = t2
            t2 = t1
            t1 = t

        assert t1 < t2 and t2 < T
        y_0 = self.processed_data[index - T + t1]
        y_t = self.processed_data[index - T + t2]
        y_T = self.processed_data[index]

        t_ = t1
        t = t2

        total_doc = item['total_doc_sentences']
        result = {
            'y_0': y_0['sentence'],
            'y_t': y_t['sentence'],
            'y_T': y_T['sentence'],
            't_': t_,
            't': t,
            'T': T,
            'total_t': total_doc,
            '0_is_author': y_0['is_author'],
            't_is_author': y_t['is_author'],
            'T_is_author': y_T['is_author']
        }

        return result

    def __len__(self):
        return len(self.processed_data)

dataset_train = SongTripletDataset(train=True, seed=13)
dataset_test = SongTripletDataset(train=False, seed=13)
dataset_train._process_data()
dataset_test._process_data()

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
dataloader_test = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

na = len(dataset_train.data["author"].unique())

author2id = {a:i for i, a in enumerate(sorted(dataset_train.data["author"].unique()))}
id2author = {i:a for a,i in author2id.items()}

model = BrownianEncoder(na, 128, 32, finetune=False)

optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

batch = next(iter(dataloader_train))

def get_loss_batch(batch, model, author2id):

    obs_0 = batch['y_0']
    obs_t = batch['y_t']
    obs_T = batch['y_T']

    is_author_0 = batch['0_is_author']
    is_author_t = batch['t_is_author']
    is_author_T = batch['T_is_author']

    t_s = batch['t_'].float()
    ts = batch['t'].float()
    Ts = batch['T'].float()

    authors_0 = [author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_0)[is_author_0.numpy()]]
    authors_t = [author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_t)[is_author_t.numpy()]]
    authors_T = [author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_T)[is_author_T.numpy()]]

    input_ids, attention_masks = dataset_train.tokenize_caption(obs_0, device)
    z_0 = model(input_ids, attention_masks, is_author_0, authors_0)

    input_ids, attention_masks = dataset_train.tokenize_caption(obs_t, device)
    z_t = model(input_ids, attention_masks, is_author_t, authors_t)

    input_ids, attention_masks = dataset_train.tokenize_caption(obs_T, device)
    z_T = model(input_ids, attention_masks, is_author_T, authors_T)

    log_q_y_T = model.get_log_q(z_t)

    loss_fn = BrownianBridgeLoss(
                z_0=z_0,
                z_t=z_t,
                z_T=z_T,
                t_=t_s,
                t=ts,
                T=Ts,
                alpha=0,
                var=0,
                log_q_y_T=log_q_y_T,
                max_seq_len=batch['total_t'].float()
            )

    loss = loss_fn.get_loss()

    return loss

def fit(epochs, model, optimizer, train_dataloader, test_dataset, author2id):

    test_dataloader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    loss_eval = 0
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        loss_training = 0
        for batch in tqdm(train_dataloader):  

            loss = get_loss_batch(batch, model, author2id)
            loss.backward()

            optimizer.step()

            loss_training+= loss.item()

        loss_training/=len(train_dataloader)

        if epoch % 5 == 0:
            model.eval()

            with torch.no_grad():
                loss_eval = 0
                for batch in tqdm(test_dataloader):

                    loss = get_loss_batch(batch, model, author2id)
                    loss_eval+= loss.item()

                loss_eval/=len(test_dataloader)
                
                for data in test_dataset.processed_data:
                    if data['is_author']:
                        data['z'] = model.authors_embeddings(torch.as_tensor(author2id[data['sentence'].replace("_AUTHOR", "")])).numpy()
                    else:
                        input_ids, attention_masks = test_dataset.tokenize_caption(data["sentence"], device)
                        data['z'] = model(input_ids, attention_masks, torch.BoolTensor([False]), torch.LongTensor([])).numpy()

                with open(os.path.join("model", "test_data_z_%d.pkl" % epoch), "wb") as f:
                    pickle.dump(test_dataset.processed_data)                    

        print("[%d/%d] Evaluation loss : %.4f  |  Training loss : %.4f" % (epoch, epochs, loss_eval, loss_training), flush=True)

        torch.save(model, os.path.join("model", "model_ckpt_%d.pt" % epoch))

fit(EPOCHS, model, optimizer, dataloader_train, dataset_test, author2id)

# tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)

# input_ids, attention_masks = dataset_train.tokenize_caption(obs_0, device)

# x = model.encoder(input_ids, attention_masks)

# distilbert_output = model.encoder(input_ids, attention_masks)
# hidden_state = distilbert_output["last_hidden_state"]

# hidden_state = hidden_state.sum(axis=1) / attention_masks.sum(axis=-1).unsqueeze(-1)

# latent_state = model.mlp(hidden_state)

# latent_state[is_author_0] = model.authors_embeddings(torch.LongTensor(authors_0))

