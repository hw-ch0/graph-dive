import json
import os
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


sentences = []


def make_sentences():
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
            sentences.append(abstract)
        f.close()


make_sentences()
tf = TfidfVectorizer(max_features=1000).fit(sentences)
print(tf.transform([sentences[0]]).toarray())
