import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import DistilBertTokenizer, BertTokenizer, GPT2Tokenizer
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score
from tqdm import tqdm
import numpy as np
from random import sample, seed
import os
import argparse
import pickle

from encoders import DISTILBERT_PATH, BrownianEncoder, MLP
from brownianlosses import BrownianBridgeLoss, BrownianLoss

from regressor import style_embedding_evaluation
from datasets import SongTripletDataset

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def set_seed(graine):
    seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)
    torch.cuda.manual_seed_all(graine)

# BATCH_SIZE = 32
# EPOCHS = 10
# LEARNING_RATE = 1e-4
# ENCODER = 'DistilBERT'
# FINETUNE = False
# CLIPNORM = 1.0
# AUTHORSPACETEXT = True
# LOSS = "BB"
# HURST = 0.9
# DATASET = "songs"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-bs','--batchsize', default=32, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('-f','--finetune', default=False, type=bool,
                        help='Finetuning of the language encoder')
    parser.add_argument('-a','--authorspace', default=False, type=bool,
                        help='Author space embedding (True for word space)')
    parser.add_argument('-lr','--learningrate', default=1e-4, type=float,
                        help='Learning rate')
    parser.add_argument('-l','--loss', default="BB", type=str,
                        help='Loss (either BB or fBM')
    parser.add_argument('-e','--encoder', default="DistilBERT", type=str,
                        help='Language encoder')
    parser.add_argument('-h','--hurst', default=1/2, type=str,
                        help='Hurst parameter (if loss is BB)')
    args = parser.parse_args()

    data_dir = args.dataset
    DATASET = data_dir.split(os.sep)[-1]
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    LEARNING_RATE = args.learningrate
    ENCODER = args.encoder
    AUTHORSPACETEXT = args.authorspace
    LOSS = args.loss
    HURST = args.hurst
    FINETUNE  = args.finetune

    CLIPNORM = 1.0

    data_dir = data_dir.split(os.sep)[-1]

    dataset_train = SongTripletDataset(data_dir = data_dir, encoder=ENCODER, train=True, seed=13)
    dataset_test = SongTripletDataset(data_dir = data_dir, encoder=ENCODER, train=False, seed=13)
    dataset_train._process_data()
    dataset_test._process_data()

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    na = len(dataset_train.data["author"].unique())

    author2id = {a:i for i, a in enumerate(sorted(dataset_train.data["author"].unique()))}
    id2author = {i:a for a,i in author2id.items()}

    model = BrownianEncoder(na, 128, 32,
                            loss = LOSS,
                            H=HURST,
                            tokenizer = ENCODER,
                            finetune = FINETUNE,
                            authorspacetxt = AUTHORSPACETEXT).to(device)

    def init_author_embeddings(init_data, author2id, model):
        print("Initializing author embedding weight")
        for author, author_id in tqdm(author2id.items()):
            input_ids, attention_mask = dataset_train.tokenize_caption(list(init_data[init_data.author==author]["sentence"]), device)
            
            model.init_author_embedding(input_ids, attention_mask, author_id)

    init_author_embeddings(dataset_train.init_data, author2id, model)

    optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

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

        if model.loss == "BB":
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
                        max_seq_len=torch.Tensor(batch['total_t'].float()).to(device),
                        H=HURST
                    )
        elif model.loss == "fBM":
            loss_fn = BrownianLoss(
                z_0=z_0,
                z_t=z_t,
                z_T=z_T,
                t_=t_s,
                t=ts,
                T=Ts,
                alpha=0,
                var=0,
                log_q_y_T=log_q_y_T,
                max_seq_len=torch.Tensor(batch['total_t'].float()).to(device),
                H=HURST
            )

        loss = loss_fn.get_loss()

        return loss

    def authorship_attribution_style_eval(model, test_dataset, author2id, epoch):

        features = pd.read_csv(os.path.join("datasets", DATASET, "features.csv"), sep=";").sort_values(by=["author", "id"])

        with torch.no_grad():

            if model.authorspacetxt:
                aut_embeddings = model.mlp(model.authors_embeddings.weight).cpu().numpy()
            else:
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

        style_df = style_embedding_evaluation(aut_embeddings, features.groupby("author").mean().reset_index(), n_fold=10)

        with open(os.path.join("results", "%s_aa_results.txt" % DATASET), "a") as f:
            f.write("%s & ce & lr \n %d & %0.2f & %0.2f" % (model.method, epoch, ce, lr))
        
        with open(os.path.join("results", "%s_style_results.txt" % DATASET), "a") as f:
            f.write("%s & style \n" % (model.method))
            f.write(style_df.transpose().to_string())

        return ce, lr, style_df

    def fit(epochs, model, optimizer, train_dataloader, test_dataset, author2id):

        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

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

                _,_,_ = authorship_attribution_style_eval(model, test_dataset, author2id, epoch)
                torch.save(model, os.path.join("model", "%s_ckpt.pt" % model.method))

                with torch.no_grad():
                    loss_eval = 0
                    for batch in tqdm(test_dataloader):

                        loss = get_loss_batch(batch, model, author2id)
                        loss_eval+= loss.item()

                    loss_eval/=len(test_dataloader)
                    
                    for data in test_dataset.processed_data:
                        if data['is_author']:
                            data['z'] = model.authors_embeddings(torch.as_tensor(author2id[data['sentence'].replace("_AUTHOR", "")]).to(device)).cpu().numpy()
                            if model.authorspacetxt:
                                data['z1'] = data['z']
                                data['z'] = model.mlp(model.authors_embeddings(torch.as_tensor(author2id[data['sentence'].replace("_AUTHOR", "")]).to(device))).cpu().numpy()
                        else:
                            input_ids, attention_masks = test_dataset.tokenize_caption(data["sentence"], device)
                            data['z'] = model(input_ids, attention_masks, torch.BoolTensor([False]).to(device), torch.LongTensor([]).to(device)).cpu().numpy()

                    with open(os.path.join("model", "%s_test_data_z.pkl" % model.method), "wb") as f:
                        pickle.dump(test_dataset.processed_data, f)                    

            print("[%d/%d] Evaluation loss : %.4f  |  Training loss : %.4f" % (epoch, epochs, loss_eval, loss_training), flush=True)

    fit(EPOCHS, model, optimizer, dataloader_train, dataset_test, author2id)