import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def text_corpus_and_year_embedding(path):
    sentences = []
    years = []
    file_list = [file for file in os.listdir(path) if file.endswith('.json')]
    for j in tqdm(file_list):
        f = open(os.path.join(path, j))
        data = json.load(f)
        try:
            abstract = data['results'][0]['abstract_inverted_index']
            year = data['results'][0]['publication_year']
            word_index = []
            for k, v in abstract.items():
                for index in v:
                    word_index.append([k, index])
                    word_index = sorted(word_index, key=lambda x: x[1])
            abstract = ""
            for k in range(len(word_index)):
                abstract += " " + str(word_index[k][0])
            sentence = data['results'][0]['title'] + ' ' + abstract
            sentences.append(sentence)
            years.append([year, year])
        except:
            pass
        f.close()
    return TfidfVectorizer(max_features=1000).fit_transform(sentences).toarray(), np.array(years)
