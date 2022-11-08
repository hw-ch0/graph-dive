import torch.utils.data as data
from utils import *


class DiveDataset(data.Dataset):
    def __init__(self, dir_path, affiliation_path, citation_threshold):
        super(DiveDataset, self).__init__()
        self.dir_path = dir_path
        self.affiliation_path = affiliation_path
        self.citation_threshold = citation_threshold
        self.texts, self.years, self.authors, self.labels = construct_data(dir_path, affiliation_path,
                                                                           citation_threshold)

    def __getitem__(self, index):
        return self.texts, self.years, self.authors, self.labels

    def __len__(self):
        return len(self.texts)
