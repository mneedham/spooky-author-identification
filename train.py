import string

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

np.set_printoptions(suppress=True)

from sklearn.pipeline import Pipeline

Y_COLUMN = "author"
TEXT_COLUMN = "text"


def test_pipeline(df, nlp_pipeline, pipeline_name=''):
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

    print("{0: <40} kfolds log losses: {1: <50}  mean log loss: {2} mean accuracy: {3}".format(
        pipeline_name,
        str([str(round(x, 3)) for x in sorted(losses)]),
        round(np.mean(losses), 3),
        round(np.mean(accuracies), 3)
    ))


train_df = pd.read_csv("train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])

tfidf_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, max_features=1500, stop_words='english')),
    ('mnb', MultinomialNB())
])

unigram_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
])

ngram_pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 3))),
    ('mnb', MultinomialNB())
])

unigram_log_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('logreg', linear_model.LogisticRegression())
])


class TextCleaner(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        df = df.apply(self.remove_punctuation)
        return df

    @staticmethod
    def remove_punctuation(text):
        for punct in string.punctuation:
            text = text.replace(punct, '')
        return text


unigram_clean_pipe = Pipeline([
    ("clean", TextCleaner()),
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
])


test_pipeline(train_df, unigram_clean_pipe, "Unigrams only (no punc pipeline)")
test_pipeline(train_df, unigram_pipe, "Unigrams only")
test_pipeline(train_df, unigram_log_pipe, "Unigrams (log reg) only")
test_pipeline(train_df, ngram_pipe, "N-grams")
test_pipeline(train_df, tfidf_pipe, "TF/IDF")


# generate output file
test_df = pd.read_csv("test.csv")

predictions = unigram_log_pipe.predict_proba(test_df["text"])

output = pd.DataFrame(predictions, columns=unigram_pipe.classes_)
output["id"] = test_df["id"]
output.to_csv("output.csv", index=False)
