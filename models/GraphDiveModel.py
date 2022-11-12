import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv
from typing import List, Tuple

class GraphDiveModel(nn.Module):
    def __init__(self,
                 text_dim: int,
                 affiliation_dim: int,
                 year_dim: int,
                 num_fc_layers: int = 2,
                 num_fc_dims: List[int] = [256, 128],
                 num_GAT_layers: int = 3,
                 num_GCN_layers: int = 2,
                 embedding_dim: int = 202,
                 hidden_dim: int = 30,
                 output_dim: int = 1,
                 heads: int = 6,
                 dropout: float = 0.3):
        super(GraphDiveModel, self).__init__()

        # FC Model
        self.text_dim = text_dim
        self.affiliation_dim = affiliation_dim
        self.year_dim = year_dim

        self.text_fc = nn.ModuleList()
        self.affiliation_fc = nn.ModuleList()

        if num_fc_layers>2:
            raise ValueError("need 2 fc layers at least.")

        for i in range(num_fc_layers):
            if i==0:
                self.text_fc.append(nn.Linear(text_dim, num_fc_dims[i]))
                self.affiliation_fc.append(nn.Linear(affiliation_dim, num_fc_dims[i]))
            elif i==(num_fc_layers-1):
                self.text_fc.append(nn.Linear(num_fc_dims[i-1],100))
                self.affiliation_fc.append(nn.Linear(num_fc_dims[i-1],100))
            else:
                self.text_fc.append(nn.Linear(num_fc_dims[i-1], num_fc_dims[i]))
                self.affiliation_fc.append(nn.Linear(num_fc_dims[i-1], num_fc_dims[i]))

        self.year_fc = nn.Linear(year_dim, 2)

        # GAT Model
        self.num_GAT_layers = num_GAT_layers
        self.heads = heads
        self.GATconvs = nn.ModuleList()
        self.GATconvs.append(self.build_GAT_model(embedding_dim, hidden_dim))
        for _ in range(num_GAT_layers - 1):
            self.GATconvs.append(self.build_GAT_model(hidden_dim, hidden_dim))

        # GCN Model
        self.num_GCN_layers = num_GCN_layers
        self.GCNconvs = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.GCNconvs.append(self.build_GCN_model(hidden_dim, hidden_dim))
        for _ in range(self.num_GCN_layers - 2):
            self.GCNconvs.append(self.build_GCN_model(hidden_dim, hidden_dim))
        self.GCNconvs.append(self.build_GCN_model(hidden_dim, hidden_dim))

        for _ in range(self.num_GCN_layers - 1):
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = dropout

    def build_GAT_model(self, input_dim, hidden_dim, concat=False):
        return GATConv(in_channels=input_dim,
                       out_channels=hidden_dim,
                       heads=self.heads,
                       concat=concat)

    def build_GCN_model(self, input_dim, hidden_dim):
        return GCNConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        text = x[:, :self.text_dim]
        affiliation = x[:, self.text_dim:self.text_dim + self.affiliation_dim]
        year = x[:, -self.year_dim:]



        for i in range(len(self.text_fc)):
            text = self.text_fc[i](text)
            text = F.relu(text)

        for i in range(len(self.affiliation_fc)):
            affiliation = self.affiliation_fc[i](affiliation)
            affiliation = F.relu(affiliation)

        year = F.relu(self.year_fc(year))

        x = torch.cat([text,affiliation,year], dim=1)
        for i in range(self.num_GAT_layers):
            x = self.GATconvs[i](x, edge_index)
            x = F.relu(x)

        for i in range(self.num_GCN_layers):
            x = self.GCNconvs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_GCN_layers - 1:
                x = self.lns[i](x)

        x = self.post_mp(x)

        return torch.sigmoid(x)

