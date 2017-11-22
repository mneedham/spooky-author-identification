import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
import math

np.set_printoptions(suppress=True)

no_features = 1000
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')

train_df = pd.read_csv("train.csv")

X = tfidf_vectorizer.fit_transform(train_df["text"])
y = train_df["author"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

logreg = linear_model.LogisticRegression(C=1e5)
# logreg.fit(X_train, y_train)
#
# y_pred = logreg.predict_proba(X_test)
# print(y_pred)
# print(metrics.log_loss(y_test, y_pred))


scores = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
print(scores.mean(), scores.std())

print(math.exp(scores.mean()))