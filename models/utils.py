import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def construct_data(dir_path:str, affiliation_path:str, citation_threshold:int):
    """
    Construct dataset for GNNs
    
    args:
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
    err_cnt = 0
    abstract_err = 0
    
    afiliation_table = pd.read_csv(affiliation_path)

    file_list = [file for file in os.listdir(dir_path) if file.endswith('.json')]
    for file in tqdm(file_list):
        f = open(os.path.join(path, file))
        paper_id = file.split('.')[0]

        affiliation_vector = afiliation_table[afiliation_table['PaperId']==paper_id].values[1:]

        try:
            data = json.load(f)
        except:
            err_cnt += 1

        abstract = data['abstract_inverted_index'] 
        if abstract==None:
            abstract_err += 1
            continue
        pub_year = data['publication_year']
        citation_cnt = data['cited_by_count']
        label = 1 if citation_cnt>citation_threshold else 0

        if pub_year < 2010:
            continue
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
        
    if not err_cnt:
        print("Warning: {} json files are failed to upload.".format(err_cnt))
    if not abstract_cnt:
        print("Warning: {} json files don't have abstract data.".format(abstract_err))
    print("{} json files are uploaded.".format(len(file_list)-err_cnt-abstract_cnt))

    return TfidfVectorizer(max_features=1000).fit_transform(sentences).toarray(), np.array(years), np.array(affiliations), np.array(labels)
