import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics
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
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        nlp_pipeline.fit(X_train, y_train)
        losses.append(metrics.log_loss(y_test, nlp_pipeline.predict_proba(X_test)))
    print(f'{pipeline_name} kfolds log losses: {str([str(round(x, 3)) for x in sorted(losses)])}')
    print(f'{pipeline_name} mean log loss: {round(pd.np.mean(losses), 3)}')

train_df = pd.read_csv("train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])
X = train_df["text"]
y = train_df["author"]

tfidf_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.95, min_df=2, max_features=1500, stop_words='english')),
    ('mnb', MultinomialNB())
                        ])
test_pipeline(train_df, tfidf_pipe, "TF/IDF")

unigram_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
                        ])
test_pipeline(train_df, unigram_pipe, "Unigrams only")

unigram_log_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', linear_model.LogisticRegression(C=1e5))
                        ])
test_pipeline(train_df, unigram_pipe, "Unigrams (log reg) only")

# predictions = model.predict_proba(test_df["text"])

# scores = cross_val_score(model, X, y, cv=10, scoring='neg_log_loss')
# print(scores.mean())


# output = test_df.copy()
# output["EAP"] = pd.Series([pred[0] for pred in predictions])
# output["HPL"] = pd.Series([pred[1] for pred in predictions])
# output["MWS"] = pd.Series([pred[2] for pred in predictions])
#
# output.to_csv("output.csv", columns=["id", "EAP", "HPL", "MWS"], index=False)