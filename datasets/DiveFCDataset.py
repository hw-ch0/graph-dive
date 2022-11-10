import torch
import numpy as np
import torch.utils.data as data
from .utils import construct_fc_data


class DiveFCDataset(data.Dataset):
    def __init__(self, dir_path, affiliation_path, citation_threshold):
        super(DiveFCDataset, self).__init__()

        self.batch_size = 128
        
        paper_ids, texts, years, authors, labels = construct_fc_data(dir_path, affiliation_path, citation_threshold)

        self.paper_ids = paper_ids
        self.data = torch.FloatTensor(np.hstack([texts, years, authors]))
        # self.data = torch.FloatTensor(np.array([texts, years, authors]))
        self.target = torch.Tensor(labels)
        

    def __getitem__(self, index):
        return self.paper_ids[index], self.data[index], self.target[index]

    def __len__(self):
        return self.data.shape[0]
    
