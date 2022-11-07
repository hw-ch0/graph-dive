import json
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def make_text_corpus_embedding(path):
    sentences = []
    file_list = [file for file in os.listdir(path) if file.endswith('.txt')]
    for j in tqdm(file_list):
        f = open(os.path.join(path, j))
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
        except:
            pass
        f.close()
    return sentences
