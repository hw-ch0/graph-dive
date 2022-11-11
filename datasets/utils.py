import os
import json
import numpy as np
import pandas as pd
import torch
import torch_geometric
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def delete_fake_node(file):
    for con in os.listdir(file):
        f = pd.read_csv(os.path.join(file, con))
        id_set = set(list(f['PaperId']) + list(f['PaperReferenceId']))
        for paper in os.listdir(os.path.join('datasets', con.split(".")[0])):
            if int(paper.split(".")[0]) not in id_set:
                os.remove(os.path.join(con.split(".")[0], paper))


def construct_fc_data(dir_path: str, affiliation_path: str, edge_data_path: str, citation_threshold: int):
    """
    args:
    Construct dataset for GNNs
    dir_path: directory of each journal/conference
    affiliation_path: CSV File path that represents authors' affiliation.
    citation_threshold: Criterion that decides whether a paper falls above or below top 10%

    return: tuple of numpy arrays
    """

    sentences = []
    years = []
    ids = []
    labels = []
    affiliations = []

    # exception case
    load_err = 0
    abstract_err = 0
    paper_id_err = 0

    affiliation_table = pd.read_csv(affiliation_path)

    # load edge_data
    edge_data = pd.read_csv(edge_data_path)
    edge_data = edge_data.iloc[:, :2]
    total_nodes = []
    total_nodes.extend(list(edge_data['PaperId']))
    total_nodes.extend(list(edge_data['PaperReferenceId']))
    total_nodes = list(set(total_nodes))

    file_list = [file for file in os.listdir(dir_path) if file.endswith('.json')]
    print("Loading json files...")
    for file in tqdm(file_list):
        f = open(os.path.join(dir_path, file))
        paper_id = int(file.split('.')[0])

        # isolated vertex->continue
        if paper_id not in total_nodes:
            continue

        try:
            affiliation_vector = affiliation_table[affiliation_table['PaperId'] == paper_id].values[0]
        except IndexError:
            paper_id_err += 1
            continue

        try:
            data = json.load(f)
        except json.JSONDecodeError:
            load_err += 1
            continue

        abstract = data['abstract_inverted_index']
        if abstract is None:
            abstract_err += 1
            continue
        pub_year = data['publication_year']
        citation_cnt = data['cited_by_count']
        label = 1 if citation_cnt > citation_threshold else 0

        year = np.zeros(13)
        year[pub_year - 2010] = 1.0
        word_index = []
        for k, v in abstract.items():
            for index in v:
                word_index.append([k, index])
                word_index = sorted(word_index, key=lambda x: x[1])
        abstract_text = ""
        for k in range(len(word_index)):
            abstract_text += " " + str(word_index[k][0])

        sentence = data['title'] + ' ' + abstract_text
        sentences.append(sentence)
        ids.append(paper_id)
        years.append(year)
        affiliations.append(affiliation_vector)
        labels.append(label)

        f.close()

    if load_err:
        print("Warning: {} json files are failed to upload.".format(load_err))
    if abstract_err:
        print("Warning: {} json files don't have abstract data.".format(abstract_err))
    if paper_id_err:
        print("Warning: {} paper-IDs do not exist.".format(paper_id_err))

    num_valid_files = len(file_list) - load_err - abstract_err - paper_id_err
    print("{}/{} json files are uploaded.".format(num_valid_files, len(file_list)))

    return np.array(ids), TfidfVectorizer(max_features=1000).fit_transform(sentences).toarray(), np.array(years), np.\
        array(affiliations), np.array(labels), num_valid_files

def construct_graph_data(paper_ids, embeddings, labels, edge_data_path:str, year_data_path:str, epoch:int) -> torch_geometric.data.Data:
    """
    epoch: for printing error
    """
    paper_ids = [i.item() for i in paper_ids]

    edge_data = pd.read_csv(edge_data_path)
    edge_data = edge_data.iloc[:,:2]

    year_data = pd.read_csv(year_data_path)
    year_data['PaperId'] = year_data['PaperId'].apply(int)

    train_idx = year_data[year_data['Year']<2018]['PaperId'].tolist()
    train_idx = [i for i in train_idx if i in paper_ids]

    val_idx = year_data[(year_data['Year']<2020) & (year_data['Year']>=2018)]['PaperId'].tolist()
    val_idx = [i for i in val_idx if i in paper_ids]

    test_idx = year_data[year_data['Year']==2020]['PaperId'].tolist()
    test_idx = [i for i in test_idx if i in paper_ids]

    node_mapping_dict = {a: i for i, a in enumerate(paper_ids)}

    edges = []
    paper_id_err = 0
    notexist_err = 0
    for idx,paper_id in enumerate(paper_ids):

        # try:
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
        print("Number of Test Nodes : {}".format(len(test_idx)))
    # data = torch_geometric.data.Data(x = torch.tensor(embeddings, dtype=torch.float),
    data = torch_geometric.data.Data(x = embeddings,
                                    #  y = torch.tensor(labels, dtype=torch.long),
                                     y = labels,
                                     edge_index = torch.tensor(edges).T,
                                     train_idx = [node_mapping_dict.get(i) for i in train_idx],
                                     val_idx = [node_mapping_dict.get(i) for i in val_idx],
                                     test_idx = [node_mapping_dict.get(i) for i in test_idx])
    return data
