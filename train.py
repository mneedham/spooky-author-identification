import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

np.set_printoptions(suppress=True)

from sklearn.pipeline import Pipeline


def create_and_fit_model(X, y):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1500, stop_words='english')
    logreg = linear_model.LogisticRegression(C=1e5)
    model = Pipeline([
        ("vectorizer", vectorizer),
        ("logreg", logreg)
    ])
    model.fit(X, y)
    return model


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

model = create_and_fit_model(train_df["text"], train_df["author"])

predictions = model.predict_proba(test_df["text"])
print(predictions)

output = test_df.copy()
output["EAP"] = pd.Series([pred[0] for pred in predictions])
output["HPL"] = pd.Series([pred[1] for pred in predictions])
output["MWS"] = pd.Series([pred[2] for pred in predictions])

output.to_csv("output.csv", columns=["id", "EAP", "HPL", "MWS"], index=False)
