import pandas as pd
from gensim.models import Word2Vec
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from util.testing import test_pipeline
from util.transformers import Word2VecFeatures

alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
lemmatizer = WordNetLemmatizer()
stop = stopwords.words('english')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = [[lemmatizer.lemmatize(word.lower())
         for word in alpha_tokenizer.tokenize(sent)
         if word.lower() not in stop]
        for sent in train.text.values]

NUM_FEATURES = 150
model = Word2Vec(data, min_count=3, size=NUM_FEATURES, window=5, sg=1, alpha=1e-4, workers=4)

pipe = Pipeline([
    ('w2v', Word2VecFeatures(model)),
    ('logreg', LogisticRegression(C=1))
])

pipe.fit(train.text, train.author.values)

# test_pipeline(train, pipe, "My pipe")

probs = pipe.predict_proba(test.text)

author = pd.DataFrame(probs)

final = pd.DataFrame()
final['id'] = test.id
final['EAP'] = author[0]
final['HPL'] = author[1]
final['MWS'] = author[2]
final.to_csv('submission3.csv', sep=',', index=False)
