"""Sarcasm-detection classifier pipelines."""

from classifiers.logreg_classifier import train as logreg_train
from classifiers.rnn_classifier import train as rnn_train
from classifiers.transformer_classifier import train as transformer_train

CLASSIFIERS = {
    "logreg": logreg_train,
    "rnn": rnn_train,
    "transformer": transformer_train,
}
