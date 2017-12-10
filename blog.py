import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

Y_COLUMN = "author"
TEXT_COLUMN = "text"


def test_pipeline(df, nlp_pipeline):
    y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])
    rskf = StratifiedKFold(n_splits=5, random_state=1)
    losses = []
    accuracies = []
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))
        accuracies.append(metrics.accuracy_score(y_test, nlp_pipeline.predict(X_test)))

    print("kfolds log losses: {0}, mean log loss: {1} mean accuracy: {2}".format(
        str([str(round(x, 3)) for x in sorted(losses)]),
        round(np.mean(losses), 3),
        round(np.mean(accuracies), 3)
    ))


unigram_log_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('logreg', linear_model.LogisticRegression())
])

ngram_pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2))),
    ('mnb', MultinomialNB())
])

classifiers = [
    ("ngram", ngram_pipe),
    ("unigram", unigram_log_pipe),
]

mixed_pipe = Pipeline([
    ("voting", VotingClassifier(classifiers, voting="soft"))
])

train_df = pd.read_csv("train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])

test_pipeline(train_df, mixed_pipe)
