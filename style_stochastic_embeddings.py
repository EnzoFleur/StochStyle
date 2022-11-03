import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from transformers import DistilBertTokenizer, BertTokenizer, GPT2Tokenizer
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
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
ENCODER = 'GPT2'
FINETUNE = False
CLIPNORM = 1.0
AUTHORSPACETEXT = False

data_dir = "datasets"

class SongTripletDataset(Dataset):

    def __init__(self, train, seed, encoder="DistilBERT", sentence_size=1):
        super(SongTripletDataset, self).__init__()

        self.train = train
        self.seed = seed
        self.max_length = 512
        self.sentence_size = sentence_size

        self.data = pd.read_csv(os.path.join(data_dir, "songs.csv"), encoding='utf-8', sep=";").sort_values(by=["author", "title"])

        if encoder == "DistilBERT":
          self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        elif encoder == "BERT":
          self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif encoder == "GPT2":
          self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
          self.tokenizer.pad_token = self.tokenizer.eos_token
          self.end_token = self.tokenizer.eos_token_id
          self.max_length = 1024

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

        self.doc_lengths = []

        self.init_data = []

        if not self.train:
          self.test_data = []

        doc_id = 0

        for author, doc in zip(self.authors, self.docs):
            
            temp = doc.split('\n')
            sentences = [ '\n'.join(temp[i:i+self.sentence_size]) for i in range(0, len(temp),self.sentence_size)]

            self.init_data.extend([{"author":author, "sentence":sentence} for sentence in sentences])
            self.doc_lengths.append(len(sentences))

            if not self.train:
                self.test_data.append(sentences)
            
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

        self.init_data = pd.DataFrame(dataset_train.init_data)

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

dataset_train = SongTripletDataset(encoder=ENCODER, train=True, seed=13)
dataset_test = SongTripletDataset(encoder=ENCODER, train=False, seed=13)
dataset_train._process_data()
dataset_test._process_data()

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

na = len(dataset_train.data["author"].unique())

author2id = {a:i for i, a in enumerate(sorted(dataset_train.data["author"].unique()))}
id2author = {i:a for a,i in author2id.items()}

model = BrownianEncoder(na, 128, 32, tokenizer = ENCODER, finetune = FINETUNE).to(device)

def init_author_embeddings(init_data, author2id, model):
    print("Initializing author embedding weight")
    for author, author_id in tqdm(author2id.items()):
        input_ids, attention_mask = dataset_train.tokenize_caption(list(init_data[init_data.author==author]["sentence"]), device)
        
        model.init_author_embedding(input_ids, attention_mask, author_id)

init_author_embeddings(dataset_train.init_data, author2id, model)

optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

# optimizer = torch.optim.SGD(params = model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

batch = next(iter(dataloader_train))

def get_loss_batch(batch, model, author2id):

    obs_0 = batch['y_0']
    obs_t = batch['y_t']
    obs_T = batch['y_T']

    is_author_0 = torch.BoolTensor(batch['0_is_author']).to(device)
    is_author_t = torch.BoolTensor(batch['t_is_author']).to(device)
    is_author_T = torch.BoolTensor(batch['T_is_author']).to(device)

    t_s = torch.Tensor(batch['t_'].float()).to(device)
    ts = torch.Tensor(batch['t'].float()).to(device)
    Ts = torch.Tensor(batch['T'].float()).to(device)

    authors_0 = torch.LongTensor([author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_0)[is_author_0.cpu().numpy()]]).to(device)
    authors_t = torch.LongTensor([author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_t)[is_author_t.cpu().numpy()]]).to(device)
    authors_T = torch.LongTensor([author2id[s.replace("_AUTHOR", "")] for s in np.array(obs_T)[is_author_T.cpu().numpy()]]).to(device)

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
                max_seq_len=torch.Tensor(batch['total_t'].float()).to(device)
            )

    loss = loss_fn.get_loss()

    return loss

def authorship_attribution_eval(model, test_dataset, author2id, epoch):
    with torch.no_grad():
        aut_embeddings = model.authors_embeddings.weight.cpu().numpy()
        doc_embeddings = []

        for doc_length, doc in zip(test_dataset.doc_lengths, test_dataset.test_data):
            input_ids, attention_masks = test_dataset.tokenize_caption(doc, device)
            doc_embedding = model(input_ids, attention_masks, torch.BoolTensor([False]*doc_length).to(device), torch.LongTensor([]).to(device)).cpu().numpy().mean(axis=0)

            doc_embeddings.append(doc_embedding)

    aut_embeddings = normalize(aut_embeddings, axis=1)
    doc_embeddings = normalize(np.vstack(doc_embeddings), axis=1)

    nd = len(doc_embeddings)

    aut_doc_test = np.zeros((nd, na))
    aut_doc_test[[i for i in range(nd)],[author2id[author] for author in test_dataset.authors]] = 1

    y_score = normalize( doc_embeddings @ aut_embeddings.transpose(),norm="l1")
    ce = coverage_error(aut_doc_test, y_score)/na*100
    lr = label_ranking_average_precision_score(aut_doc_test, y_score)*100

    with open(os.path.join("results", "aa_results.txt"), "w") as f:
        f.write("%s & ce & lr \n %d & %03f & %03f" % (model.method, epoch, ce, lr))
    
    return ce, lr

def fit(epochs, model, optimizer, train_dataloader, test_dataset, author2id):

    test_dataloader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    loss_eval = 0
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()

        loss_training = 0
        for batch in tqdm(train_dataloader):  

            loss = get_loss_batch(batch, model, author2id)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIPNORM)
            optimizer.step()

            loss_training+= loss.item()

        loss_training/=len(train_dataloader)

        if epoch % 10 == 0:
            model.eval()

            ce, lr = authorship_attribution_eval(model, test_dataset, author2id, epoch)
            torch.save(model, os.path.join("model", "model_ckpt_%d.pt" % epoch))

            with torch.no_grad():
                loss_eval = 0
                for batch in tqdm(test_dataloader):

                    loss = get_loss_batch(batch, model, author2id)
                    loss_eval+= loss.item()

                loss_eval/=len(test_dataloader)
                
                for data in test_dataset.processed_data:
                    if data['is_author']:
                        data['z'] = model.authors_embeddings(torch.as_tensor(author2id[data['sentence'].replace("_AUTHOR", "")]).to(device)).cpu().numpy()
                    else:
                        input_ids, attention_masks = test_dataset.tokenize_caption(data["sentence"], device)
                        data['z'] = model(input_ids, attention_masks, torch.BoolTensor([False]).to(device), torch.LongTensor([]).to(device)).cpu().numpy()

                with open(os.path.join("model", "test_data_z_%d.pkl" % epoch), "wb") as f:
                    pickle.dump(test_dataset.processed_data, f)                    

        print("[%d/%d] Evaluation loss : %.4f  |  Training loss : %.4f" % (epoch, epochs, loss_eval, loss_training), flush=True)

fit(EPOCHS, model, optimizer, dataloader_train, dataset_test, author2id)