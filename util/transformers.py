import string
from collections import Counter, defaultdict

import nltk
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
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


# Number of words in the text
# Number of unique words in the text
# Number of characters in the text
# Number of stopwords
# Number of punctuations
# Number of upper case words
# Number of title case words
# Average length of the words


eng_stopwords = set(stopwords.words("english"))


class MetaFeatures(TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, text_series):
        df = pd.DataFrame(text_series)
        df["num_words"] = text_series.apply(self.num_words)
        df["unique_words"] = text_series.apply(self.num_unique_words)
        df["num_characters"] = text_series.apply(self.num_chars)
        df["num_stopwords"] = text_series.apply(self.num_stopwords)
        df["num_words_title"] = text_series.apply(self.num_title_words)
        df["num_punctuation"] = text_series.apply(self.num_punctuation)
        df["ave_word_length"] = text_series.apply(self.ave_word_length)

        return df

    @staticmethod
    def num_words(sentence):
        return len(str(sentence).split())

    @staticmethod
    def num_unique_words(sentence):
        return len(set(str(sentence).split()))

    @staticmethod
    def ave_word_length(sentence):
        return np.mean([len(w) for w in str(sentence).split()])

    @staticmethod
    def num_punctuation(sentence):
        return len([c for c in str(sentence) if c in string.punctuation])

    @staticmethod
    def num_chars(sentence):
        return len(sentence)

    @staticmethod
    def num_title_words(sentence):
        return len([w for w in str(sentence).split() if w.istitle()])

    @staticmethod
    def num_stopwords(sentence):
        return len([w for w in str(sentence).lower().split() if w in eng_stopwords])


class Word2VecFeatures(TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.alpha_tokenizer = nltk.RegexpTokenizer('[A-Za-z]\w+')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop = stopwords.words('english')
        self.NUM_FEATURES = 150

    def fit(self, x, y=None):
        return self

    def transform(self, text_series):

        vectors = []
        for i in text_series:
            values = self.get_feature_vec(
                [self.lemmatizer.lemmatize(word.lower()) for word in self.alpha_tokenizer.tokenize(i) if
                 word.lower() not in self.stop], self.NUM_FEATURES, self.model)
            vectors.append(values)

        return np.array(vectors)

    @staticmethod
    def get_feature_vec(tokens, num_features, model):
        featureVec = np.zeros(shape=(1, num_features), dtype='float32')
        missed = 0
        for word in tokens:
            try:
                featureVec = np.add(featureVec, model[word])
            except KeyError:
                missed += 1
                pass
        if len(tokens) - missed == 0:
            return np.zeros(shape=num_features, dtype='float32')
        return np.divide(featureVec, len(tokens) - missed).squeeze()
