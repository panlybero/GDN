# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc,get_prior_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset
from models.FeatureGrouper import CorrelationProblem, get_feature_grouper

from models.GDN import GDN
from models.MaskedGDN import MaskedGDN
from models.AnomalyTransformer import AnomalyTransformer

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random
import pickle


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)
       
        train, test = train_orig, test_orig
        
        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        if '2B_AIT_002_PV' in train.columns and False:
            train = train.drop(columns=['2B_AIT_002_PV'])
            test = test.drop(columns=['2B_AIT_002_PV'])

        feature_map = get_feature_map(dataset)
        #fc_struc = get_prior_graph_struc(dataset)#get_fc_graph_struc(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

       

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
            'custom_edges':env_config['custom_edges']
        }

        edge_index = fc_edge_index
        test_edge_index = fc_edge_index

        if env_config['custom_edges']:
            
            print("running with custom edges")
            with open("./data/gridworlds/train_edges.pkl",'rb') as f:
                edge_index = pickle.load(f)
            with open("./data/gridworlds/test_edges.pkl",'rb') as f:
                test_edge_index = pickle.load(f)

        train_dataset = TimeDataset(train_dataset_indata, edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, test_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        #Create masked model or regular model

        print(train_config)
        if train_config['n_masks']>len(feature_map):
            raise ValueError(f"Can't have more masks that features {train_config['n_masks']}>{len(feature_map)}")

        if train_config['n_masks']>0:
            indeces = np.arange(len(feature_map))
            perm = indeces#np.random.permutation(indeces)
            
            
            if train_config['n_masks']!=len(feature_map):
                #Get groups using grouper
                print((self.train_dataset.x.shape))
                
                grouper = get_feature_grouper("ClusteringGrouper",train_config=train_config,train_dataset=self.train_dataset,initial_state=perm)
                groups = grouper.get_groups()
            else:
                groups =[[i] for i in indeces]

            self.model = MaskedGDN(train_config['n_masks'],edge_index_sets, len(feature_map), 
                    dim=train_config['dim'], 
                    input_dim=train_config['slide_win'],
                    out_layer_num=train_config['out_layer_num'],
                    out_layer_inter_dim=train_config['out_layer_inter_dim'],
                    topk=train_config['topk'],
                    batch = train_config['batch'],
                    masking_indeces = groups
                ).to(self.device)
            self.score_func = get_full_err_scores
            print(groups)
            print("Made model")
        else:
            if train_config['model']=='GDN':
                self.model = GDN(edge_index_sets, len(feature_map), 
                        dim=train_config['dim'], 
                        input_dim=train_config['slide_win'],
                        out_layer_num=train_config['out_layer_num'],
                        out_layer_inter_dim=train_config['out_layer_inter_dim'],
                        topk=train_config['topk']
                    ).to(self.device)
            elif train_config['model']=='Transformer':
                self.model = AnomalyTransformer(len(feature_map),train_config['slide_win'],1,3).to(self.device)




    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]
            self.model_save_path = model_save_path
            self.train_log = train(self.model, model_save_path, 
                config = self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset'],
                lr=self.train_config['lr']
            )
        
        # test

        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)
        
        test_loss, self.test_result = test(best_model, self.test_dataloader)
        val_loss, self.val_result = test(best_model, self.val_dataloader)

        test_per_entry_loss = (np.array(self.test_result[0]) - np.array(self.test_result[1]))**2
        val_per_entry_loss = (np.array(self.val_result[0]) - np.array(self.val_result[1]))**2

        anomaly_dict = dict(test = test_per_entry_loss.T, normal = val_per_entry_loss.T)
        self.anomaly_dict = anomaly_dict

        results_dict_loss=self.get_score(self.test_result, self.val_result, anomaly_scores=anomaly_dict)
        results_dict_loss = {k+"_loss":v for k,v in results_dict_loss.items()}
        results_dict=self.get_score(self.test_result, self.val_result)
        
        results_dict.update(results_dict_loss)
        results_dict['val_loss']=val_loss
        results_dict['test_loss']=test_loss


        for k,v in results_dict.items():
            print(k,":",v)

        return results_dict

        

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result, anomaly_scores = None):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        if anomaly_scores is None:
            test_scores, normal_scores = get_full_err_scores(test_result, val_result)
            print(test_scores.shape, normal_scores.shape)
        else:
            test_scores, normal_scores = anomaly_scores['test'],anomaly_scores['normal']

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1) 
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        
        metrics = ['F1',"precision",'recall','AUC']
        results_dict = {m:info[i] for i,m in enumerate(metrics)}
        return results_dict


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-custom_edges', help='use_custom_edges', type = str, default='false')
    parser.add_argument('-n_masks', help='how many masks to use', type = int, default=0)
    parser.add_argument('-group_search', help='how long to search for good groups', type = float, default=0)
    parser.add_argument('-model', help='Which model to use, GDN or Transformer', type = str, default="Transformer")
    parser.add_argument('-lr',type=float, default=1e-3)
    args = parser.parse_args()

    #random.seed(args.random_seed)
    #np.random.seed(args.random_seed)
    #torch.manual_seed(args.random_seed)
    #torch.cuda.manual_seed(args.random_seed)
    #torch.cuda.manual_seed_all(args.random_seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True
    #os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    import types
    #myargs = types.SimpleNamespace(batch=32, comment='swat', custom_edges='false', dataset='swat', decay=0, device='cuda:0', dim=64, epoch=50, group_search=1, load_model_path='', model='GDN', n_masks=50, out_layer_inter_dim=16, out_layer_num=2, random_seed=0, report='best', save_path_pattern='swat', slide_stride=1, slide_win=5, topk=15, val_ratio=0.2,lr=1e-3)
    #args = myargs

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'n_masks':args.n_masks,
        'group_search':args.group_search,
        'model':args.model,
        'lr':args.lr
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'custom_edges': True if args.custom_edges == 'true' else False
    }
    print(env_config)

    main = Main(train_config, env_config, debug=False)
    main.run()




