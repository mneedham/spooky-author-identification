from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def estimator_spec_for_softmax_classification(logits, labels, mode):
    """Returns EstimatorSpec instance for softmax classification."""
    predicted_classes = tf.argmax(logits, 1)
    # predicted_classes = logits
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': tf.nn.softmax(logits),
                'prob': tf.nn.softmax(logits),
                'log_loss': tf.nn.softmax(logits),
            })
    print("labels: {0}".format(labels))
    print("logits: {0}".format(logits))
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # print(tf.one_hot(labels, depth=MAX_LABEL), logits)
    # loss = tf.losses.log_loss(labels=tf.one_hot(labels, depth=MAX_LABEL), predictions=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
    """A bag-of-words model. Note it disregards the word order in the text."""
    bow_column = tf.feature_column.categorical_column_with_identity(WORDS_FEATURE, num_buckets=n_words)
    bow_embedding_column = tf.feature_column.embedding_column(bow_column, dimension=EMBEDDING_SIZE)
    bow = tf.feature_column.input_layer(features, feature_columns=[bow_embedding_column])
    logits = tf.layers.dense(bow, MAX_LABEL, activation=None)
    return estimator_spec_for_softmax_classification(logits=logits, labels=labels, mode=mode)

tf.logging.set_verbosity(tf.logging.INFO)

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0
MAX_LABEL = 3
WORDS_FEATURE = 'words'  # Name of the input words feature.

Y_COLUMN = "author"
TEXT_COLUMN = "text"

train_df = pd.read_csv("train.csv")

X = pd.Series(train_df[TEXT_COLUMN])


y = train_df[Y_COLUMN].copy()

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

test_df = pd.read_csv("test.csv")

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

X_transform_train = vocab_processor.fit_transform(X_train)
X_transform_test = vocab_processor.transform(X_test)

X_train = np.array(list(X_transform_train))
X_test = np.array(list(X_transform_test))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

model_fn = bag_of_words_model
classifier = tf.estimator.Estimator(model_fn=model_fn)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: X_train},
    y=y_train,
    batch_size=len(X_train),
    num_epochs=None,
    shuffle=True)
classifier.train(input_fn=train_input_fn, steps=100)

# Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class'] for p in predictions))
print(y_predicted)
y_predicted = y_predicted.reshape(np.array(y_test).shape)


# Score with sklearn.
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

# Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print(scores)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
