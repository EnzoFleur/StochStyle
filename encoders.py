import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False

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
            ])
        
        def forward(self, x):
            return self.mlp(x)

class BrownianEncoder(nn.Module):
    def __init__(self, na, hidden_dim, latent_dim, tokenizer="DistilBERT", finetune=False):
        super(BrownianEncoder, self).__init__()

        self.na = na
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune
        self.tokenizer = tokenizer

        if self.tokenizer == "DistilBERT":
          self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        elif self.tokenizer == "BERT":
          self.encoder = BertModel.from_pretrained("bert-base-uncased")
        elif self.tokenizer == "GPT2":
          self.encoder = GPT2Model.from_pretrained("gpt2")
        
        for param in self.encoder.parameters():
            param.requires_grad = self.finetune

        self.mlp = MLP(768, self.hidden_dim, self.latent_dim)

        self.log_q = self.create_log_q()

        self.C_eta = nn.Linear(1, 1)

        # Switch off bias in linear layers
        self.mlp.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

        self.authors_embeddings = nn.Embedding(self.na, 32)

    def init_author_embedding(self, input_ids, attention_mask, author_id):
        with torch.no_grad():
            encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = encoder_output[0]

            hidden_state = hidden_state.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)

            latent_state = self.mlp(hidden_state.mean(axis=0))

            self.authors_embeddings.weight[author_id] = latent_state

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
                               ])

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def get_log_q(self, x):
        return self.log_q(x)

    def forward(self, input_ids, attention_mask, is_author, authors):

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, attention_mask)

        latent_state = self.mlp(hidden_state)

        latent_state[is_author] = self.authors_embeddings(authors)

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