import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from util.testing import Y_COLUMN, TEXT_COLUMN, test_pipeline
from util.transformers import MetaFeatures, DropStringColumns

train_df = pd.read_csv("train.csv", usecols=[Y_COLUMN, TEXT_COLUMN])

# print(train_df.head())

meta_pipe = Pipeline([
    ('meta', MetaFeatures()),
    ("dropStrings", DropStringColumns()),
    # ("scale", Normalizer()),
    ('mnb', MultinomialNB())
])

# meta_pipe.fit(train_df[TEXT_COLUMN], train_df[Y_COLUMN])
# df = meta_pipe.transform(train_df[TEXT_COLUMN])
# print(df)


# df = meta_pipe._transform(train_df[TEXT_COLUMN])
# print(df.head())

test_pipeline(train_df, meta_pipe, "Mixed")