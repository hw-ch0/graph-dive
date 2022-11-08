import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class FCModel(nn.Module):
    def __init__(self, training=True):
        super(FCModel, self).__init__()
        self.textfc = nn.Linear(1000, 100)
        # self.authorfc = nn.Linear(???, 100)
        self.yearfc = nn.Linear(13, 2)
        self.training = training

    def forward(self, data):
        text_emb, year_emb = text_corpus_and_year_embedding(data)
        text_emb = self.textfc(text_emb)
        # author_emb = self.authorfc(author_emb)
        year_emb = self.yearfc(year_emb)
        # return torch.cat([text_emb, author_emb, year_emb])
        return torch.cat([text_emb, year_emb])
