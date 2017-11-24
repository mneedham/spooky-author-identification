import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

np.set_printoptions(suppress=True)

train_df = pd.read_csv("train.csv")
y = train_df["author"]
logreg = linear_model.LogisticRegression(C=1e5)

# work out how many tokens we should keep
token_range = [10, 50, 75, 100, 300, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]
token_scores = []
for tokens in token_range:
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=tokens, stop_words='english')
    X = vectorizer.fit_transform(train_df["text"])
    scores = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')
    token_scores.append(abs(scores.mean()))

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.plot(token_range, token_scores)
plt.xlabel('# of tokens')
plt.ylabel('Cross-Validated Accuracy')

plt.show()