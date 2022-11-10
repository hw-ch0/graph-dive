import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def construct_fc_data(dir_path:str, affiliation_path:str, citation_threshold:int):
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
    citation = []
    labels = []
    affiliations = []

    # exception case
    load_err = 0
    abstract_err = 0
    paper_id_err = 0

    affiliation_table = pd.read_csv(affiliation_path)

    file_list = [file for file in os.listdir(dir_path) if file.endswith('.json')]
    for file in tqdm(file_list):
        f = open(os.path.join(dir_path, file))
        paper_id = file.split('.')[0]

        try:
            affiliation_vector = affiliation_table[affiliation_table['PaperId']==int(paper_id)].values[0]
        except IndexError:
            paper_id_err += 1
            continue

        try:
            data = json.load(f)
        except json.JSONDecodeError:
            load_err += 1
            continue

        abstract = data['abstract_inverted_index'] 
        if abstract==None:
            abstract_err += 1
            continue
        pub_year = data['publication_year']
        citation_cnt = data['cited_by_count']
        label = 1 if citation_cnt>citation_threshold else 0

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

    print("{} json files are uploaded.".format(len(file_list)-load_err-abstract_err-paper_id_err))

    return TfidfVectorizer(max_features=1000).fit_transform(sentences).toarray(), np.array(years), np.array(affiliations), np.array(labels)
