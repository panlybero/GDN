import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AnomalyAttention(nn.Module):
    def __init__(self, seq_dim, in_channels, out_channels):
        super(AnomalyAttention, self).__init__()
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.W_sigma= nn.Linear(in_channels, 1, bias=False)

        self.Q = self.K = self.V = self.sigma = torch.zeros((seq_dim, out_channels))
        self.d_model = out_channels
        self.n  = seq_dim
        self.P = torch.zeros((seq_dim, seq_dim))
        self.S = torch.zeros((seq_dim, seq_dim))

    def forward(self, x):

        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()

        return Z

    def initialize(self, x):
        # self.d_model = x.shape[-1]
        self.Q = self.W_q(x)
        self.K = self.W_k(x)
        self.V = self.W_v(x)
        self.sigma = self.W_sigma(x)
        
        
        

    def prior_association(self):
        p = torch.from_numpy(
            np.abs(
                np.indices((self.n,self.n))[0] - 
                np.indices((self.n,self.n))[1]
                )
            ).to(self.sigma.data.get_device())
        p = p.repeat((self.sigma.shape[0],1,1))
        
        gaussian = torch.normal(p.float(), self.sigma.abs())
        
        gaussian /= gaussian.sum(dim=-1).unsqueeze(-1)
        
        return gaussian

    def series_association(self):
        return F.softmax((torch.bmm(self.Q,torch.transpose(self.K,1,2))) / math.sqrt(self.d_model), dim=0)

    def reconstruction(self):
        return torch.bmm(self.S, self.V)

    def association_discrepancy(self):
        return F.kl_div(self.P, self.S) + F.kl_div(self.S, self.P)


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, seq_dim, feat_dim):
        super().__init__()
        self.seq_dim, self.feat_dim = seq_dim, feat_dim
       
        self.attention = AnomalyAttention(self.seq_dim, self.feat_dim, self.feat_dim)
        self.ln1 = nn.LayerNorm(self.feat_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU()
        )
        self.ln2 = nn.LayerNorm(self.feat_dim)
        self.association_discrepancy = None

    def forward(self, x):
        x_identity = x 
        x = self.attention(x)
        z = self.ln1(x + x_identity)
        
        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        self.association_discrepancy = self.attention.association_discrepancy().detach()
        
        return z

class AnomalyTransformer(nn.Module):
    def __init__(self, seqs, in_channels, layers, lambda_):
        super().__init__()
        self.blocks = nn.ModuleList([
            AnomalyTransformerBlock(seqs, in_channels) for _ in range(layers)
        ])
        self.output = None
        self.lambda_ = lambda_
        self.assoc_discrepancy = torch.zeros((seqs, len(self.blocks)))
        self.seqs= seqs
    
    def forward(self, x,edges, **kwargs):
        self.assoc_discrepancy = torch.zeros((self.seqs, len(self.blocks)))
        for idx, block in enumerate(self.blocks):
            x = block(x)
            self.assoc_discrepancy[:, idx] = block.association_discrepancy
        
        self.assoc_discrepancy = self.assoc_discrepancy.sum(dim=-1) #N x 1
        self.output = x
        return x

    def loss_func(self, x,last_state):
        
        l2_norm = torch.norm(self.output - x, p=2)
        return l2_norm + (self.lambda_ * self.assoc_discrepancy.mean())

    def anomaly_score(self, x, edges, **kwargs):
        out = self(x,edges,**kwargs)
        score = F.softmax(-self.assoc_discrepancy, dim=0) 
        return score
    def test_prediction(self,x, edges, **kwargs):
        return self.anomaly_score(x, edges, **kwargs)