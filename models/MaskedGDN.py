import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from .utils import get_divisors
from .graph_layer import GraphLayer
from .GDN import GDN

class MaskedGDN(nn.Module):
    '''
    Wrapper Module for GDN that implements masking.
    
    '''
    def __init__(self, n_masks,edge_index_sets, node_num,**kwargs):
        super().__init__()
        kwargs['input_dim']+=1 # +1 to include last state
        self.input_dim = kwargs['input_dim'] # includes last state
        self.node_num = node_num
        self.GDN = GDN(edge_index_sets,node_num,**kwargs)
        self.masking_indeces = kwargs['masking_indeces']
        self.n_masks = n_masks
        self.feat_order = [n for g in self.masking_indeces for n in g]
        
        print(f"Using masked model with {n_masks} masks on {node_num} features.")
        self.masks = self._create_masks(self.node_num,n_masks,kwargs['batch'])

    def _create_masks(self,n_feats, n_masks,bsz = 1):
        '''
        Generates masks that are used for prediction. We could have at most N masks for N nodes. We can also choose M masks such that each node is only masked once.

        '''
        feat_groups = self.masking_indeces
        masks = torch.ones(bsz,n_masks,n_feats,1)
        for i,group in enumerate(feat_groups):
            masks[:,i,group]=0

        return masks

    def loss_func(self,y_pred, y_true):
        loss = F.mse_loss(y_pred, y_true, reduction='mean')
        return loss

    def forward(self, data, org_edge_index, last_state):
        #create masked version of last state
        n_masks = self.n_masks
        n_features = self.node_num
        
        masks = self.masks.to(data.get_device())[:last_state.shape[0]].detach()

        masked_last_state = last_state.reshape(-1,1,n_features,1) * masks #batch_size,num_masks,n_nodes,1

        x = data.reshape([-1, 1,self.node_num, self.input_dim-1])
        repeated_x = x.expand(x.shape[0],n_masks,x.shape[2],x.shape[3]) #shape: (bsz,n_masks,n_nodes,n_steps)

        x_with_masked_last_state = torch.cat((repeated_x,masked_last_state), axis =-1)
        x_with_masked_last_state = x_with_masked_last_state.view(-1,self.node_num,self.input_dim)

        data = x_with_masked_last_state # shape: (bsz*n_masks,n_nodes,n_feats)
       
        out = self.GDN(data,org_edge_index)

        out = out.reshape(-1,n_masks,n_features,1)  #split by mask
        out = out[masks==0].view(-1,self.node_num) #collect the predictions for masked values

        #feat_order =#torch.cat(torch.tensor(self.masking_indeces)).to(out.get_device())
        result = torch.zeros_like(out)
        result[:,self.feat_order]= out

        
        #print(result)
        return result

    def test_prediction(self,data, org_edge_index, last_state):
        return self(data,org_edge_index,last_state)