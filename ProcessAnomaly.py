'''
Online process anomaly detection using word2vec encoding and autocloud

pipeline:
1. creation of a list containing all traces, each trace is a list with its activities (as strings);
2. word2vec model trained with previously created list;
3. creation of an average vector for each trace. this is done by averaging the word vectors from the trace;
4. train test split and lof fitted using the average vectors;
5. classification and metrics computation.
'''

import os
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from sklearn import metrics
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import gensim
from gensim.models import Word2Vec
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

n_workers = cpu_count()


def read_log(path, log):
    '''
    Reads event log and preprocess it
    '''
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['event_processed'] = df_raw['activity_name'].str.replace(' ', '-')
    labels = [1 if x == 'normal' else -1 for x in df_raw['label']]
    df_raw['label'] = labels
    df_proc = df_raw[['case_id', 'event_processed', 'label']]
    del df_raw
    return df_proc


def read_window(df_raw):
    '''
    Reads event log and preprocess it
    '''
    df_raw['event_processed'] = df_raw['activity_name'].str.replace(' ', '-')
    labels = [1 if x == 'normal' else -1 for x in df_raw['label']]
    df_raw['label'] = labels
    df_proc = df_raw[['case_id', 'event_processed', 'label']]
    del df_raw
    return df_proc


def cases_y_list(df):
    '''
    Creates a list of cases for model training
    '''
    cases, y, case_id = [], [], []
    for group in df.groupby('case_id'):
        events = list(group[1].event_processed)
        cases.append([''.join(x) for x in events])
        y.append(list(group[1].label)[0])
        case_id.append(group[0])

    return cases, y, case_id


def create_models(cases, size, window, min_count):
    '''
    Creates a word2vec model
    '''
    model = Word2Vec(
                size=size,
                window=window,
                min_count=min_count,
                workers=n_workers)
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)

    return model


#def create_models(cases, size, window, min_count, tipo, old_model):
    #    '''
    #Creates a word2vec model
    #'''
    #if tipo:
    #    model = Word2Vec(
    #        size=size,
    #        window=window,
    #        min_count=min_count,
    #        workers=n_workers)
    #    model.build_vocab(cases)
    #    model.train(cases, total_examples=len(cases), epochs=10)
    #    model.wv.vocab
    #    model.save("old_model")

    #   return model


    #else:
    #    new_model = Word2Vec.load("old_model")
    #    new_model.build_vocab(cases, update=True)
    #    new_model.train(cases, total_examples=2, epochs=1)
    #    new_model.wv.vocab

    #   return new_model


def average_feature_vector(cases, model):
    '''
    Computes average feature vector for each trace
    '''
    vectors = []
    for case in cases:
        case_vector = []
        for token in case:
            try:
                case_vector.append(model.wv[token])
            except KeyError:
                pass
        vectors.append(np.array(case_vector).mean(axis=0))

    return vectors


def compute_metrics(y_true, y_pred):
    '''
    Computes performance metrics
    '''
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)

    return acc, f1, precision, recall
