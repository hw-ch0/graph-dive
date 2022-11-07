import json
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def make_text_corpus_embedding():
    x = 0
    y = 0
    sentences = []
    file_list = os.listdir('./paper')
    for j in tqdm(file_list):
        f = open('./paper/' + j)
        data = json.load(f)
        try:
            abstract = data['results'][0]['abstract_inverted_index']
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
            x += 1
        except:
            y += 1
            pass
        f.close()
    return sentences
