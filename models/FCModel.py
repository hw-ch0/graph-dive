import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self, text_dim, affiliation_dim, year_dim):
        super(FCModel, self).__init__()
        
        self.text_dim = text_dim
        self.affiliation_dim = affiliation_dim
        self.year_dim = year_dim

        self.text_fc = nn.ModuleList()
        self.text_fc.append(nn.Linear(text_dim, 256))
        self.text_fc.append(nn.Linear(256, 100))

        self.affiliation_fc = nn.ModuleList()
        self.affiliation_fc.append(nn.Linear(affiliation_dim, 256))
        self.affiliation_fc.append(nn.Linear(256, 100))

        self.year_fc = nn.Linear(year_dim, 2)

    def forward(self, data):
        # text, affiliation, year = data
        text = data[:,:self.text_dim]
        affiliation = data[:,self.text_dim:self.text_dim+self.affiliation_dim]
        year = data[:,-self.year_dim:]

        for i in range(len(self.text_fc)):
            text = self.text_fc[i](text)
            text = F.relu(text)

        for i in range(len(self.affiliation_fc)):
            affiliation = self.affiliation_fc[i](affiliation)
            affiliation = F.relu(affiliation)

        year = F.relu(self.year_fc(year))

        return torch.cat([text, affiliation, year], dim=1)