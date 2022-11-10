import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.3, training=True):
        super(GCNModel, self).__init__()

        self.num_layers = num_layers #2
        self.convs = nn.ModuleList()
        self.lns = nn.ModuleList()

        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for _ in range(self.num_layers-2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        for _ in range(self.num_layers-1):
            self.lns.append(nn.LayerNorm(hidden_dim))

        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.dropout = dropout
        self.training = training

    def build_conv_model(self, input_dim, hidden_dim):
        return GCNConv(input_dim, hidden_dim)

    def forward(self, x, data):
        x, edge_index, batch = x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.post_mp(x)

        return x

    def loss(self, pred, label):
        return F.cross_entropy(pred, label)