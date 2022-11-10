import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, heads):
        super(GATModel, self).__init__()

        self.num_layers = num_layers #3
        self.heads = heads

        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for _ in range(num_layers-2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.convs.append(self.build_conv_model(hidden_dim, output_dim))

    def build_conv_model(self, input_dim, hidden_dim, dropout=0.1, concat=False):
        return GATConv(in_channels=input_dim,
                       out_channels=hidden_dim,
                       heads=self.heads,
                       concat=concat)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        return x