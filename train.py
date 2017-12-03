from pprint import pprint
from time import time

import numpy as np
import pandas as pd
from polyglot.text import Text
from sklearn import linear_model
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from util.testing import test_pipeline, Y_COLUMN, TEXT_COLUMN
from util.transformers import TextCleaner, PoSScanner, DropStringColumns
from util.output import generate_output_file

np.set_printoptions(suppress=True)

train_df = pd.read_csv("train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])

tfidf_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=3, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english')),
    ('mnb', MultinomialNB())
])

unigram_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
])

ngram_pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1, 2))),
    ('mnb', MultinomialNB())
])

unigram_log_pipe = Pipeline([
    ('cv', CountVectorizer()),
    ('logreg', linear_model.LogisticRegression())
])

unigram_clean_pipe = Pipeline([
    ("clean", TextCleaner()),
    ('cv', CountVectorizer()),
    ('mnb', MultinomialNB())
])

pos_pipe = Pipeline([
    ("clean", TextCleaner()),
    ("pos", PoSScanner()),
    ("dropStrings", DropStringColumns()),
    ("logreg", linear_model.LogisticRegression())
])

entities = {}


def analyze(doc):
    if doc not in entities:
        entities[doc] = ["_".join(entity) for entity in Text(doc, hint_language_code="en").entities]
    return entities[doc]


nlp_pipeline = Pipeline([
    ('cv', CountVectorizer(analyzer=analyze)),
    ('mnb', MultinomialNB())
])

classifiers = [
    ("tfidf", tfidf_pipe),
    ("ngram", ngram_pipe),
    ("unigram", unigram_log_pipe),
    ("nlp", nlp_pipeline)
]

mixed_pipe = Pipeline([
    ("voting", VotingClassifier(classifiers, voting="soft"))
])


def combinations_on_off(num_classifiers):
    return [[int(x) for x in list("{0:0b}".format(i).zfill(num_classifiers))]
            for i in range(1, 2 ** num_classifiers)]


if __name__ == "__main__":
    # test_pipeline(train_df, unigram_clean_pipe, "Unigrams only (no punc pipeline)")
    # test_pipeline(train_df, unigram_pipe, "Unigrams only")
    # test_pipeline(train_df, pos_pipe, "PoS (log reg) only")

    # test_pipeline(train_df, unigram_log_pipe, "Unigrams (log reg) only")
    # test_pipeline(train_df, ngram_pipe, "N-grams")
    # test_pipeline(train_df, tfidf_pipe, "TF/IDF")
    # test_pipeline(train_df, mixed_pipe, "Mixed")
    # generate_output_file(mixed_pipe)

    # should be able to generate all the variations of the estimators
    # should do a separate grid search on each of the individual classifiers?

    param_grid = dict(
        voting__weights=combinations_on_off(len(classifiers)),
    )
    grid_search = GridSearchCV(mixed_pipe, param_grid=param_grid, n_jobs=-1, verbose=10, scoring="neg_log_loss")

    df = pd.read_csv("train.csv")
    y = df[Y_COLUMN].copy()
    X = pd.Series(df[TEXT_COLUMN])

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in mixed_pipe.steps])
    print("parameters:")
    pprint(param_grid)
    t0 = time()
    grid_search.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
