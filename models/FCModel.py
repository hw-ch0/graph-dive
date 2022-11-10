import torch
import torch.nn as nn
import torch.nn.functional as F


class FCModel(nn.Module):
    def __init__(self, text_dim, affiliation_dim, year_dim):
        super(FCModel, self).__init__()
        self.text_fc = nn.Linear(text_dim, 100)
        self.affiliation_fc = nn.Linear(affiliation_dim, 100)
        self.year_fc = nn.Linear(year_dim, 2)

    def forward(self, data):
        text, affiliation, year = data
        text_emb = F.relu(self.text_fc(text))
        affiliation_emb = F.relu(self.affiliation_fc(affiliation))
        year_emb = F.relu(self.year_fc(year))

        return torch.cat([text_emb, affiliation_emb, year_emb])