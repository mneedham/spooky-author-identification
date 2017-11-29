import pandas as pd
import numpy as np
import scipy.sparse as sp
import array

from polyglot.text import Text
from collections import defaultdict

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

vocabulary = defaultdict()
vocabulary.default_factory = vocabulary.__len__

df = pd.read_csv("train.csv")

j_indices = []
indptr = array.array(str("i"))
values = array.array(str("i"))
indptr.append(0)

for t in df.text:
    feature_counter = {}
    entities = Text(t, hint_language_code="en").entities

    for entity in entities:
        feature = "_".join(entity)
        feature_idx = vocabulary[feature]

        if feature_idx not in feature_counter:
            feature_counter[feature_idx] = 1
        else:
            feature_counter[feature_idx] += 1

    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    indptr.append(len(j_indices))

j_indices = np.asarray(j_indices, dtype=np.intc)
indptr = np.frombuffer(indptr, dtype=np.intc)
values = np.frombuffer(values, dtype=np.intc)

vocabulary = dict(vocabulary)
X = sp.csr_matrix((values, j_indices, indptr),
                  shape=(len(indptr) - 1, len(vocabulary)),
                  dtype=np.int64)

np.set_printoptions(threshold=np.nan)

Y_COLUMN = "author"
TEXT_COLUMN = "text"

nlp_pipeline = MultinomialNB()
pipeline_name = "entity_extraction"

y = df[Y_COLUMN].copy()
# X = pd.Series(df[TEXT_COLUMN])
rskf = StratifiedKFold(n_splits=5, random_state=1)
losses = []
accuracies = []
for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    nlp_pipeline.fit(X_train, y_train)
    losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))
    accuracies.append(metrics.accuracy_score(y_test, nlp_pipeline.predict(X_test)))

print("{0: <40} kfolds log losses: {1: <50}  mean log loss: {2} mean accuracy: {3}".format(
    pipeline_name,
    str([str(round(x, 3)) for x in sorted(losses)]),
    round(np.mean(losses), 3),
    round(np.mean(accuracies), 3)
))





