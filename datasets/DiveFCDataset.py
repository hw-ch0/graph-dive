import torch.utils.data as data
from .utils import construct_fc_data


class DiveFCDataset(data.Dataset):
    def __init__(self, dir_path, affiliation_path, citation_threshold):
        super(DiveFCDataset, self).__init__()

        self.batch_size = 128
        
        texts, years, authors, labels = construct_data(dir_path, affiliation_path, citation_threshold)
        self.data = torch.FloatTensor([texts, years, authors])
        self.target = torch.FloatTensor(labels)
        

    def __getitem__(self, index):
        return self.texts, self.years, self.authors, self.labels
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.target)
