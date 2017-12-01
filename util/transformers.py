import string
from collections import Counter, defaultdict

import nltk
import pandas as pd
from sklearn.base import TransformerMixin


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


class PoSScanner(TransformerMixin):
    @staticmethod
    def count_pos(pos_to_count, text):
        c = Counter()
        for term, pos in nltk.pos_tag(nltk.word_tokenize(text)):
            c[pos] += 1
        return float(c[pos_to_count])

    def fit(self, x, y=None):
        return self

    def transform(self, text_series):
        df = pd.DataFrame(text_series)

        new_columns = defaultdict(list)
        for text in text_series:
            c = Counter()
            for term, pos in nltk.pos_tag(nltk.word_tokenize(text)):
                c[pos] += 1

            for pos in ["NN", "VB", "VBG", "UH", "RBR", "RBS", "PRP", "NNS", "NNP", "JJS", "IN", "CC"]:
                new_columns[pos].append(c[pos])

        for pos in new_columns.keys():
            df["{0}_count".format(pos)] = new_columns[pos]

        return df


class DropStringColumns(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, df):
        for col, dtype in zip(df.columns, df.dtypes):
            if dtype == object:
                del df[col]
        return df
