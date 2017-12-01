import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

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
