import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


year_range = 122
min_year = 1900
# year_range, min_year = find_year_range([...])


def find_year_range(path_list):
    max_year = 0
    min_year = 3000
    for path in path_list:
        file_list = [file for file in os.listdir(path) if file.endswith('.json')]
        for file in tqdm(file_list):
            f = open(os.path.join(path, file))
            data = json.load(f)
            try:
                year = data['results'][0]['publication_year']
                max_year = max(year, max_year)
                min_year = min(year, min_year)
            except:
                pass
    return max_year - min_year, min_year


def text_corpus_and_year_embedding(path):
    sentences = []
    years = []
    file_list = [file for file in os.listdir(path) if file.endswith('.json')]
    for file in tqdm(file_list):
        f = open(os.path.join(path, file))
        data = json.load(f)
        try:
            abstract = data['results'][0]['abstract_inverted_index']
            pub_year = data['results'][0]['publication_year']
            year = np.zeros(year_range)
            year[pub_year - min_year] = 1.0
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
            years.append(year)
        except:
            pass
        f.close()
    return TfidfVectorizer(max_features=1000).fit_transform(sentences).toarray(), np.array(years)
