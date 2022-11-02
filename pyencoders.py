import torch.nn as nn
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")


class DAN(nn.Module):

    def __init__(self, input_dim, hidden, r):
        super(DAN, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.r = r

        self.do1 = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.do2 = nn.Dropout(0.1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, r)

    def forward(self, x):

        x = x.mean(dim=1)
        x = self.do1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.do2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x

class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(MLP, self).__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim

            self.mlp = nn.Sequential(*[
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Dropout(0.1),
            ])
        
        def forward(self, x):
            return self.mlp(x)

class BrownianEncoder(nn.Module):
    def __init__(self, na, hidden_dim, latent_dim, finetune=False):
        super(BrownianEncoder, self).__init__()

        self.na = na
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune

        self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        for param in self.encoder.parameters():
            param.requires_grad = self.finetune

        self.mlp = MLP(768, self.hidden_dim, self.latent_dim)

        self.log_q = self.create_log_q()

        self.C_eta = nn.Linear(1, 1)

        self.authors_embeddings = nn.Embedding(self.na, 32)
        nn.init.normal_(self.authors_embeddings.weight, mean=0.0, std=0.02)

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
                               ])

    def get_log_q(self, x):
        return self.log_q(x)

    def forward(self, input_ids, attention_masks, is_author, authors):

        distilbert_output = self.encoder(input_ids, attention_masks)
        hidden_state = distilbert_output["last_hidden_state"]

        hidden_state = hidden_state.sum(axis=1) / attention_masks.sum(axis=-1).unsqueeze(-1)

        latent_state = self.mlp(hidden_state)

        latent_state[is_author] = self.authors_embeddings(torch.LongTensor(authors))

        return latent_state


class BrownianBridgeLoss(object):
    """Everything is a brownian bridge...

    p(z_t | mu_0, mu_T) = \mathcal{N}(mu_0 * t/T + mu_T * (1-t/T), I t*(T-t)/T)

    normalization constant: -1/(2 * t*(T-t)/T)
    """

    def __init__(self,
                z_0, z_t, z_T,
                t_, t, T,
                alpha, var,
                log_q_y_T,
                max_seq_len,
                eps=1e-6,
                C_eta=None,
                label=None):
        super().__init__()
        self.log_q_y_T = log_q_y_T
        self.z_0 = z_0
        self.z_t = z_t
        self.z_T = z_T
        self.t_ = t_
        self.t = t
        self.T = T
        self.alpha = alpha
        self.var = var
        self.loss_f = self.simclr_loss
        self.eps= eps
        self.max_seq_len = max_seq_len
        self.sigmoid = nn.Sigmoid()
        self.label = label

        if C_eta is None:
            C_eta = 0.0
        self.C_eta = C_eta
        self.end_pin_val = 1.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _log_p(self, z_0, z_t, z_T, t_0, t_1, t_2):
        T = t_2-t_0
        t = t_1-t_0

        alpha = (t/(T+self.eps)).view(-1, 1)
        delta = z_0 * (1-alpha) + z_T * (alpha) - z_t
        var = (t * (T - t)/ (T + self.eps))
        log_p =  -1/(2*var + self.eps) * (delta*delta).sum(-1) + self.C_eta # (512,)
        if len(log_p.shape) > 1: # (1, bsz)
            log_p = log_p.squeeze(0)
        return log_p

    def _logit(self, z_0, z_T, z_t, t_, t, T):
        """
        Calculating log p(z_tp1, z_t) = -|| h(z_{t+dt}) - h(z_t)(1-dt)||^2_2
        """
        log_p = self._log_p(z_0=z_0, z_t=z_t, z_T=z_T,
                            t_0=t_, t_1=t, t_2=T)
        log_p = log_p.unsqueeze(-1)
        log_q = self.log_q_y_T
        logit = log_p # - log_q
        return logit # should be (bsz, 1)

    def reg_loss(self):
        loss = 0.0
        mse_loss_f = nn.MSELoss()
        # start reg
        start_idxs = torch.where((self.t_) == 0)[0]
        if start_idxs.nelement():
            vals = self.z_0[start_idxs, :]
            start_reg = mse_loss_f(vals, torch.zeros(vals.shape, device=self.device))
            loss += start_reg
        # end reg
        end_idxs = torch.where((self.T) == self.max_seq_len - 1)[0]
        if end_idxs.nelement():
            vals = torch.abs(self.z_T[end_idxs, :])
            end_reg = mse_loss_f(vals, torch.ones(vals.shape, device=self.device)*self.end_pin_val)
            loss += end_reg
        return loss

    def simclr_loss(self):
        """
        log p = -1/(2*eta) \| x' - x - \mu(x) \|^2_2 + C_{\eta}

        logit = log p - log q
        """
        loss = 0.0
        # Positive pair
        pos_logit = self._logit(z_0=self.z_0, z_T=self.z_T, z_t=self.z_t,
                                t_=self.t_, t=self.t, T=self.T)
        pos_probs = torch.exp(pos_logit) # (bsz,1)
        for idx in range(self.z_T.shape[0]):
            # Negative pair: logits over all possible contrasts
            # Nominal contrast for random triplet - contrast from in between
            neg_i_logit = self._logit(
                z_0=self.z_0, z_T=self.z_T, z_t=self.z_t[idx],
                t_=self.t_, t=self.t[idx], T=self.T)
            neg_i_probs = torch.exp(neg_i_logit) # (bsz,1)
            loss_i = -(pos_logit[idx] - torch.log(neg_i_probs.sum() + self.eps))
            loss += loss_i

        loss = loss / self.z_T.shape[0]
        # Regularization for pinning start and end of bridge
        reg_loss = self.reg_loss()
        loss += reg_loss
        return loss

    def get_loss(self):
        return self.loss_f()
