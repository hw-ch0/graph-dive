import json
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def make_sentences():
    sentences = []
    file_list = os.listdir('./paper')
    for j in tqdm(file_list):
        f = open('./paper/' + j)
        data = json.load(f)
        for i in range(len(data['results'])):
            abstract = data['results'][i]['abstract_inverted_index']
            word_index = []
            try:
                for k, v in abstract.items():
                    for index in v:
                        word_index.append([k, index])
                        word_index = sorted(word_index, key=lambda x: x[1])
            except:
                pass
            abstract = ""
            for k in range(len(word_index)):
                abstract += " " + str(word_index[k][0])
            sentence = data['results'][i]['title'] + ' ' + abstract
            sentences.append(sentence)
        f.close()
    return sentences
