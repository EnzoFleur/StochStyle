import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, BertTokenizerFast, GPT2TokenizerFast
from sklearn.model_selection import train_test_split
import os
import re
from tqdm import tqdm

import random

from nltk.tokenize import sent_tokenize

from encoders import DISTILBERT_PATH, BERT_PATH, GPT2_PATH

############# Text Reader ###############
def clean_str(string):
    string= re.sub(r"[^A-Za-z0-9!\"\£\€#$%\&\’'()\*\+,\-.\/:;\<\=\>?\@[\]\^\_`{\|}\~\n]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f_in:
        content=clean_str(f_in.read())
    return(content)

class SongTripletDataset(Dataset):

    def __init__(self, data_dir, train, seed, author_mode=1, encoder="DistilBERT", sentence_size=1):
        super(SongTripletDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        self.max_length = 512
        self.sentence_size = sentence_size

        # Author_mode is either 1 (only added at the start) or 2 (added at the start and the end of the document)
        self.author_mode = author_mode

        self.data = pd.read_csv(os.path.join(data_dir, "songs.csv"), encoding='utf-8', sep=";").sort_values(by=["author", "title"])

        if encoder == "DistilBERT":
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
            self.tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
        elif encoder == "GPT2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_PATH)
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
            doc_length = len(sentences) + self.author_mode

            self.doc_lengths.append(doc_length)

            if not self.train:
                self.test_data.append(sentences)
            
            # Add the author as starting point of the document
            self.processed_data.append({
                "sentence": "%s_AUTHOR" % author,
                "sentence_id": 0,
                "doc_id": doc_id,
                "total_doc_sentences": doc_length,
                "is_author": True
            })

            for sentence_id, sentence in enumerate(sentences, start=1):
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "doc_id": doc_id,
                    "total_doc_sentences": doc_length,
                    "is_author": False
                }
                self.processed_data.append(sentence_info)

            if self.author_mode == 2:
                # Add the author as ending point of the document
                self.processed_data.append({
                    "sentence": "%s_AUTHOR" % author,
                    "sentence_id": sentence_id+1,
                    "doc_id": doc_id,
                    "total_doc_sentences": doc_length,
                    "is_author": True
                })

            doc_id += 1

        print("Length of dataset: %d" % len(self.processed_data))
        print("Number of documents: %d" % (doc_id))

        self.init_data = pd.DataFrame(self.init_data)

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


class GutenbergTripletDataset(Dataset):

    def __init__(self, data_dir, train, seed, author_mode=1, encoder="DistilBERT", sentence_size=1):
        super(GutenbergTripletDataset, self).__init__()

        self.data_dir = data_dir
        self.train = train
        self.seed = seed
        self.max_length = 512
        self.sentence_size = sentence_size
        self.author_mode = author_mode

        self.authors = []
        self.docs = []
        unique_authors = sorted([a for a in os.listdir(os.path.join(data_dir)) if os.path.isdir(os.path.join(data_dir, a))])
        for author in tqdm(unique_authors):
            docs = sorted([doc for doc in os.listdir(os.path.join(data_dir, author))])

            for doc in docs:
                content = read(os.path.join(data_dir, author, doc))
                self.authors.append(author)
                self.docs.append(content)

        if encoder == "DistilBERT":
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_PATH)
        elif encoder == "BERT":
            self.tokenizer = BertTokenizerFast.from_pretrained(BERT_PATH)
        elif encoder == "GPT2":
            self.tokenizer = GPT2TokenizerFast.from_pretrained(GPT2_PATH)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.end_token = self.tokenizer.eos_token_id
            self.max_length = 1024

        if self.train:
            self.authors, _, self.docs, _= train_test_split(
                                        self.authors, self.docs,
                                        test_size=0.2,
                                        stratify=self.authors,
                                        random_state=self.seed
                                        )
        else:
            _, self.authors, _, self.documents = train_test_split(
                            self.authors, self.docs,
                            test_size=0.2,
                            stratify=self.authors,
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
            
            sentences = sent_tokenize(doc)

            self.init_data.extend([{"author":author, "sentence":sentence} for sentence in sentences])
            doc_length = len(sentences) + self.author_mode

            self.doc_lengths.append(doc_length)

            if not self.train:
                self.test_data.append(sentences)
            
            # Add the author as starting point of the document
            self.processed_data.append({
                "sentence": "%s_AUTHOR" % author,
                "sentence_id": 0,
                "doc_id": doc_id,
                "total_doc_sentences": doc_length,
                "is_author": True
            })

            for sentence_id, sentence in enumerate(sentences, start=1):
                sentence_info = {
                    "sentence": sentence,
                    "sentence_id": sentence_id,
                    "doc_id": doc_id,
                    "total_doc_sentences": doc_length,
                    "is_author": False
                }
                self.processed_data.append(sentence_info)

            if self.author_mode == 2:
                # Add the author as ending point of the document
                self.processed_data.append({
                    "sentence": "%s_AUTHOR" % author,
                    "sentence_id": sentence_id+1,
                    "doc_id": doc_id,
                    "total_doc_sentences": doc_length,
                    "is_author": True
                })

            doc_id += 1

        print("Length of dataset: %d" % len(self.processed_data))
        print("Number of documents: %d" % (doc_id))

        self.init_data = pd.DataFrame(self.init_data)

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