import itertools
import numpy as np
from simanneal import Annealer
from .utils import get_divisors 


def get_feature_grouper(name='AnnealingGrouper',**kwargs):
    if name =='AnnealingGrouper':
        return AnnealingGrouper(**kwargs)
    if name =='ClusteringGrouper':
        return ClusteringGrouper(**kwargs)

class  AnnealingGrouper:
    def __init__(self,**kwargs):
        train_config = kwargs['train_config']
        train_dataset = kwargs['train_dataset']
        crp  = CorrelationProblem(train_config['n_masks'],train_dataset.x[:,:,0],initial_state = kwargs['initial_state'])
        print("Scheduling....",flush=True)
        auto_schedule = crp.auto(minutes=train_config['group_search']) 
        crp.set_schedule(auto_schedule)
        print("Searching...",flush= True)
        perm, cost = crp.anneal()
        print("Done...",flush= True)
        self.groups = crp.segment(perm)

        pass
    def get_groups(self):
        return self.groups


class ClusteringGrouper:
    def __init__(self,**kwargs):
        from sklearn.cluster import SpectralClustering
        train_config = kwargs['train_config']
        train_dataset = kwargs['train_dataset']
        data_matrix = train_dataset.x[:,:,0]
        self.n_masks = train_config['n_masks']
        corrs = np.corrcoef(data_matrix,rowvar=False)
        corrs[np.isnan(corrs)==1]=0
        weight_matrix = 1-np.abs(corrs) # highly correlated get small weights

        clst = SpectralClustering(self.n_masks,affinity='precomputed').fit(weight_matrix)
        self.labels = clst.labels_

	    

    def get_groups(self):
        groups = [[] for i in range(self.n_masks)]
        for i,l in enumerate(self.labels):
            groups[l].append(i)
        
	    #ensure all groups have assignments
        groups = list(filter(lambda x: len(x)>0, groups))
        return groups



class CorrelationProblem(Annealer):
    def __init__(self, n_masks,data_matrix, **kwargs) -> None:
        super().__init__(**kwargs)
        corrs = np.corrcoef(data_matrix,rowvar=False)
        corrs[np.isnan(corrs)==1]=0
        self.corrs = corrs
        self.n_masks = n_masks
        self.nodes_per_group = [[] for i in range(n_masks)]
        msk_idx = 0
        for i in range(corrs.shape[0]):
            self.nodes_per_group[msk_idx].append(i)
            msk_idx = (msk_idx+1)%n_masks


    def move(self):
        """Swap two indeces"""
        tmp = self.segment(self.state)
        masks = np.random.choice(np.arange(self.n_masks),size=(2,), replace = False)
        inds = np.random.choice(np.arange(len(tmp[masks[0]])),size=(1,), replace = True)[0],np.random.choice(np.arange(len(tmp[masks[1]])),size=(1,), replace = True)[0]
        #print(inds)
        q = tmp[masks[0]][inds[0]]
        tmp[masks[0]][inds[0]] = tmp[masks[1]][inds[1]]
        tmp[masks[1]][inds[1]] = q
        self.state = [i for sub in tmp for i in sub]
        
        
    def energy(self):
        """Calculates the length of the route."""
        
        return self.calculate_perm_cost(self.state)


    def calculate_total_corrs(self,group):
        if len(group)==1:
            return 0
        pairs = np.array(list(itertools.combinations(group,2)), int)
        
        s = np.abs(self.corrs[pairs[:,0],pairs[:,1]]).sum()
        return s
        
    def segment(self,perm):
        
        parts = [[perm[i] for i in group] for group in self.nodes_per_group]
        #nodes_per_mask = len(perm)//self.n_masks
        #remainder = len(perm) - nodes_per_mask*self.n_masks
        #parts = [perm[i:i+nodes_per_mask] for i in range(0,len(perm)-remainder,nodes_per_mask)] + [perm[len(perm)-remainder:]]
        
        return parts

    def calculate_perm_cost(self,perm):
        segments = self.segment(perm)
        costs = []
        for seg in segments:
            #print(seg)
            costs.append(self.calculate_total_corrs(seg))
            
        return sum(costs)
