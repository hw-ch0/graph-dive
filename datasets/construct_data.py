import os
import json
import numpy as np
import pandas as pd
import torch
import torch_geometric
from .utils import construct_fc_data
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def construct_data(dir_path: str,
                   affiliation_path: str,
                   edge_data_path: str,
                   year_data_path: str,
                   citation_threshold: int,
                   epoch: int=0,
                   train_window_size: int=5,
                   val_window_size: int=1,
                   start_year: int = 2011) -> torch_geometric.data.Data:
    """
    epoch: for printing error
    """
    if start_year<2011 or start_year>(2021-train_window_size-val_window_size):
        raise ValueError("invalid start year")

    paper_ids, texts, years, affiliations, labels, num_valid_files = construct_fc_data(dir_path, affiliation_path, edge_data_path, citation_threshold)

    paper_ids = [i.item() for i in paper_ids]

    edge_data = pd.read_csv(edge_data_path)
    edge_data = edge_data.iloc[:,:2]

    year_data = pd.read_csv(year_data_path)
    year_data = year_data[year_data['Year']<(start_year+train_window_size+val_window_size)]
    year_data['PaperId'] = year_data['PaperId'].apply(int)

    train_idx = year_data[year_data['Year']<(start_year+train_window_size)]['PaperId'].tolist()
    train_idx = [i for i in train_idx if i in paper_ids]

    val_idx = year_data[year_data['Year']>=(start_year+train_window_size)]['PaperId'].tolist()
    val_idx = [i for i in val_idx if i in paper_ids]

    node_mapping_dict = {a: i for i, a in enumerate(paper_ids)}

    edges = []
    paper_id_err = 0
    notexist_err = 0
    for idx,paper_id in enumerate(paper_ids):

        try:
            tmp = year_data[year_data['PaperId']==paper_id].values[0][0]
        except IndexError:
            continue
        try:
            src = edge_data[edge_data['PaperId']==paper_id].values[0][0]
            tgt = edge_data[edge_data['PaperId']==paper_id].values[0][1]
        except IndexError:
            paper_id_err += 1
            continue

        mapped_src = node_mapping_dict.get(src)
        mapped_tgt = node_mapping_dict.get(tgt)
        if mapped_src==None or mapped_tgt==None:
            notexist_err += 1
            continue

        if mapped_src in train_idx:
            edges.append([mapped_src, mapped_tgt])
            edges.append([mapped_tgt, mapped_src])
        else:
            edges.append([mapped_tgt, mapped_src])

    if epoch==0:
        print("Warning: {} Paper Ids don't exist in edge data".format(paper_id_err))
        print("Number of Train Nodes : {}".format(len(train_idx)))
        print("Number of Validation Nodes : {}".format(len(val_idx)))
    # data = torch_geometric.data.Data(x = torch.tensor(embeddings, dtype=torch.float),
    data = torch_geometric.data.Data(x = torch.FloatTensor(np.hstack([texts, years])),
                                     y = torch.Tensor(labels),
                                     edge_index = torch.tensor(edges).T,
                                     train_idx = [node_mapping_dict.get(i) for i in train_idx],
                                     val_idx = [node_mapping_dict.get(i) for i in val_idx])
    return data
