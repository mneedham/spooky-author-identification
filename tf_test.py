from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

EMBEDDING_SIZE = 50
MAX_LABEL = 3
WORDS_FEATURE = 'words'  # Name of the input words feature.


def bag_of_words_model(features, labels, mode):
    bow_column = tf.feature_column.categorical_column_with_identity(WORDS_FEATURE, num_buckets=n_words)
    bow_embedding_column = tf.feature_column.embedding_column(bow_column, dimension=EMBEDDING_SIZE)
    bow = tf.feature_column.input_layer(features, feature_columns=[bow_embedding_column])
    logits = tf.layers.dense(bow, MAX_LABEL, activation=None)
    return create_estimator_spec(logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for softmax
    # classification over output classes.
    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return create_estimator_spec(logits=logits, labels=labels, mode=mode)


def create_estimator_spec(logits, labels, mode):
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits),
                'log_loss': tf.nn.softmax(logits),
            })

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


tf.logging.set_verbosity(tf.logging.INFO)

Y_COLUMN = "author"
TEXT_COLUMN = "text"
le = preprocessing.LabelEncoder()

train_df = pd.read_csv("train.csv")
X = pd.Series(train_df[TEXT_COLUMN])
y = le.fit_transform(train_df[Y_COLUMN].copy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

MAX_DOCUMENT_LENGTH = 100
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)

X_transform_train = vocab_processor.fit_transform(X_train)
X_transform_test = vocab_processor.transform(X_test)

X_train = np.array(list(X_transform_train))
X_test = np.array(list(X_transform_test))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

model_fn = rnn_model
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

scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy: {0:f}, Loss {1:f}'.format(scores['accuracy'], scores["loss"]))

# output
test_df = pd.read_csv("test.csv")

X_test = pd.Series(test_df[TEXT_COLUMN])
X_test = np.array(list(vocab_processor.transform(X_test)))

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={WORDS_FEATURE: X_test},
    num_epochs=1,
    shuffle=False)

predictions = classifier.predict(test_input_fn)
y_predicted_classes = np.array(list(p['prob'] for p in predictions))

output = pd.DataFrame(y_predicted_classes, columns=le.classes_)
output["id"] = test_df["id"]
output.to_csv("output.csv", index=False, float_format='%.6f')
