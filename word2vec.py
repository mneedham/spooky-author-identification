import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(sent) if word.lower() not in stop] for
        sent in train.text.values]

NUM_FEATURES = 150

model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)

print(len(model.wv.vocab))

print(model.most_similar('raven'))


def get_feature_vec(tokens, num_features, model):
    featureVec = np.zeros(shape=(1, num_features), dtype='float32')
    missed = 0
    for word in tokens:
        try:
            featureVec = np.add(featureVec, model[word])
        except KeyError:
            missed += 1
            pass
    if len(tokens) - missed == 0:
        return np.zeros(shape=num_features, dtype='float32')
    return np.divide(featureVec, len(tokens) - missed).squeeze()


vectors = []
for i in train.text.values:
    vectors.append(get_feature_vec(
        [lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop],
        NUM_FEATURES, model))

train['author'] = train['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})

estimator = LogisticRegression(C=1)
estimator.fit(np.array(vectors), train.author.values)

test_vectors = []
for i in test.text.values:
    test_vectors.append(get_feature_vec(
        [lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(i) if word.lower() not in stop],
        NUM_FEATURES, model))

probs = estimator.predict_proba(test_vectors)

author = pd.DataFrame(probs)

final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
final.to_csv('submission.csv', sep=',', index=False)
