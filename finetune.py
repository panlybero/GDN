import itertools
from main import Main
import types
import itertools
import copy
import numpy as np
import threading
import torch
from multiprocessing.pool import Pool
import torch.multiprocessing as mp
import pandas as pd
from slack_message import send_message
import json
import os
def run_config(args):
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
    print(train_config,env_config)

    main = Main(train_config, env_config, debug=False)
    results_dict = main.run()
    return results_dict, main.model_save_path


class ConfigGenerator:
    
    '''
    Generator for inividual configs. Given a namespace with the possible values per variable, creates namespaces with individual instanciations of the variables
    '''
    def __init__(self,args):
        self.args= args
        self.arg_ranges = {}
        for k,vars in args.__dict__.items():
            if isinstance(vars,list):
                self.arg_ranges[k]=vars
    
        
        self.all_combinations = list(itertools.product(*list(self.arg_ranges.values())))
    
    def combination_to_args(self, comb):
        keys = list(self.arg_ranges.keys())
        arg_dict = {k:v for k,v in zip(keys,comb)}
        return arg_dict
    
    def __getitem__(self,index):
        args = types.SimpleNamespace(**copy.deepcopy(self.args.__dict__))
        comb_dict = self.combination_to_args(self.all_combinations[index])
        for k,v in comb_dict.items():
            args.__dict__[k]=v
        return args
        
    def __len__(self):
        return len(self.all_combinations)
    

if __name__=='__main__':


    args = types.SimpleNamespace()
    args.batch=32
    args.device='cuda:2'
    args.epoch = 50
    args.slide_win=5
    args.dim=list(range(16,129,16))
    args.slide_stride=1
    args.dataset='swat'
    args.comment=args.dataset
    args.random_seed=np.random.randint(10000)
    args.out_layer_num = [1,2,3,4]
    args.out_layer_inter_dim=list(range(16,129,16))
    args.decay=0
    args.topk=[5,10,15,20]
    args.n_masks=50
    args.group_search=1
    args.model='GDN'
    args.save_path_pattern=args.dataset
    args.report='best'
    args.load_model_path=''
    args.custom_edges = 'false'
    args.val_ratio = 0.2
    args.lr = [1e-3,1e-4]

    torch.multiprocessing.set_start_method('spawn') 
    configs = ConfigGenerator(args)
    print(len(configs))
    random_configs = np.random.choice(range(len(configs)), 200,replace=False)
    
    random_configs = [configs[i] for i in random_configs]
    result_list = {}
    for indx,cfg in enumerate(random_configs):
        
        procs = []
        result,save_path=run_config(cfg)
        '''
        with Pool(processes=4) as pool: 
            confs = []
            for i in range(4):
                conf = copy.deepcopy(cfg)
                conf.device = f"cuda:{i}"
                conf.seed=i
                confs.append(conf)
            try:
                results = pool.map_async(run_config,confs)
                results = results.get()
            except RuntimeError:
                print("oom error, reducing batch size to 16")
                torch.cuda.empty_cache()
                for conf in confs:
                    conf.batch=16
                results = pool.map_async(run_config,confs)
                results = results.get()
        '''
        #df = pd.DataFrame(results)
        #mean = df.mean(axis=0)
        
        #result_dict={"params":str(cfg),**dict(mean)}        
        result_dict={"params":str(cfg),**result}      
        send_message(json.dumps(result_dict)) 
        result_list[save_path]=copy.deepcopy(result_dict)

        if 'finetuning_results.json' in os.listdir('./'):
            with open('finetuning_results.json','r') as f:
                existing_results = json.load(f)

            result_list.update(existing_results)

        with open('finetuning_results.json','w') as f:
                json.dump(result_list,f)
            

    send_message('Done finetuning')


        
        




    
