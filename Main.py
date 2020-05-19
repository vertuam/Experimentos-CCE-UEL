import sys, os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import re
import time
import psutil
import pandas as pd
import numpy as np
import ProcessAnomaly as pa
import AutoCloud as ac
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from model.Case import Case
import nltk
import matplotlib.pyplot as plt
import math
import Outlier as ot
import pickle
from math import inf

# hyperparameters configuration
path = 'data'
process = 'sample_data.csv'
auto_cloud_m = 2
type_of_pre_pross = 'Autocloud'  # 'Word2Vec', 'text_to_numbers', 'mahalanobis', 'Autocloud'
log_type = 2
parameters = [(60, 700), (40, 600), (50, 500), (100, 400), (50, 300)]

start_time = time.time()
event_stream = pd.read_csv(f'{path}/{process}')
event_stream = event_stream.values
scl = StandardScaler()
dados = np.array([])


def text_to_numbers(text, cutoff_for_rare_words=1):
    """Function to convert text to numbers. Text must be tokenzied so that
    test is presented as a list of words. The index number for a word
    is based on its frequency (words occuring more often have a lower index).
    If a word does not occur as many times as cutoff_for_rare_words,
    then it is given a word index of zero. All rare words will be zero.
    """

    # Flatten list if sublists are present
    if len(text) > 1:
        flat_text = [item for sublist in text for item in sublist]
    else:
        flat_text = text

    # get word freuqncy
    fdist = nltk.FreqDist(flat_text)

    # Convert to Pandas dataframe
    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
    df_fdist.columns = ['Frequency']

    # Sort by word frequency
    df_fdist.sort_values(by=['Frequency'], ascending=False, inplace=True)

    # Add word index
    number_of_words = df_fdist.shape[0]
    df_fdist['word_index'] = list(np.arange(number_of_words) + 1)

    # replace rare words with index zero
    frequency = df_fdist['Frequency'].values
    word_index = df_fdist['word_index'].values
    mask = frequency <= cutoff_for_rare_words
    word_index[mask] = 0
    df_fdist['word_index'] = word_index

    # Convert pandas to dictionary
    word_dict = df_fdist['word_index'].to_dict()

    # Use dictionary to convert words in text to numbers
    text_numbers = []
    for string in text:
        string_numbers = [word_dict[word] for word in string]
        text_numbers.append(string_numbers)

    return (text_numbers)


# Usa Word2Vec
if type_of_pre_pross == 'Word2Vec':
    # reads event log
    df = pa.read_log(path, process)
    # process cases and labels
    cases, y, case_id = pa.cases_y_list(df)
    del df
    # generate model -> cases, size, window, min_count
    model = pa.create_models(cases, 250, 1, 1)
    # calculating the average feature vector for each sentence (trace)
    vectors = pa.average_feature_vector(cases, model)
    # normalization
    vectors = scl.fit_transform(vectors)

    # Teste com Distancia
    x = np.array([pt[0] for pt in vectors])
    y = np.array([pt[1] for pt in vectors])
    teste = np.sqrt(np.square(x - x.reshape(-1, 1)) + np.square(y - y.reshape(-1, 1)))

    # recebe dados
    data = pd.DataFrame(vectors)
    # chamar autocloud
    dados = data.T
    print('Numero de Dimensoes: ', dados.ndim)
elif type_of_pre_pross == 'text_to_numbers':
    # reads event log
    df = pa.read_log(path, process)
    # process cases and labels
    cases, y, case_id = pa.cases_y_list(df)
    del df
    print("Total of Cases: ", np.size(cases))
    opa = text_to_numbers(cases)
    data = pd.DataFrame(opa)
    dados = np.array(
        [data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[6], data[8], data[9], data[10],
         data[11], data[12], data[13], data[14]])

    # dados = np.where(np.isnan(dados), 0, dados)

    # Para chamar autocloud
    # dados = dados.T

    print('Numero de Dimensoes: ', dados.ndim)

    new_dados = np.empty((0, 41))

    # result_array = np.empty((0, 41))
    for i in range(0, np.size(dados)):
        if i >= 4999:
            break
        result_array = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0])
        novo_array = dados[i]
        for j in range(0, np.size(novo_array)):
            if not math.isnan(novo_array[int(j)]):
                k = novo_array[int(j)]
                result_array[int(k - 1)] += np.long(999000)
        new_dados = np.append(new_dados, [result_array], axis=0)
        dados = new_dados
elif type_of_pre_pross == 'mahalanobis':
    # reads event log
    df = pa.read_log(path, process)
    # process cases and labels
    cases, y, case_id = pa.cases_y_list(df)
    # generate model -> cases, size, window, min_count
    model = pa.create_models(cases, 250, 1, 1)
    # calculating the average feature vector for each sentence (trace)
    vectors = pa.average_feature_vector(cases, model)
    # normalization
    vectors = scl.fit_transform(vectors)
    i = 1
    PrecisionList = []
    RecallList = []
    N = len(vectors)
    Y = df.iloc[:, -1]
    del df
    distancematrix = ot.createDistanceMatrix(vectors, ot.first_time, N)
    # O is the #of outliers
    for (k, O) in parameters:
        print("Experiment:", i, ", k =", k, ", num_outliers =", O)
        lrd = ot.getLRD(N, distancematrix, k, vectors)
        sorted_outlier_factor_indexes = np.argsort(lrd)
        outliers = sorted_outlier_factor_indexes[-O:]
        ot.getAccuracy(outliers, Y, N, PrecisionList, RecallList)
        i += 1
elif type_of_pre_pross == 'Autocloud':
    # Carregar Data set
    # data = pd.read_csv('data/dim1024.txt', sep='\s+', header=None)
    # data = pd.read_csv('data/array.txt', sep=',', header=None)
    data = pd.read_csv('data/dim512.txt', sep='\s+', header=None)
    # data = pd.read_csv('data/a1.csv', sep='\s+', header=None)
    # data = pd.read_csv('data/s1.csv', sep='\s+', header=None)
    # Carregar para Array A1 e S1
    dados = np.array([data[0], data[1]])
    # dados = np.array([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12],data[13],data[14]])
    dados = dados.T

# Inicio AutoCloud
teste = ac.AutoCloud(auto_cloud_m)
# for t in X_embedded:
if type_of_pre_pross == 'Autocloud':
    for t in dados:
        teste.run(np.array(t), 'na')
else:
    for i in range(0, len(dados), 1):
     teste.run(dados[i], case_id[i])

print('Alfa: ', teste.alfa)
print('listIntersection: ', teste.listIntersection)
print('relevanceList: ', teste.relevanceList)

print('Numero de Clouds: ', np.size(teste.c))

print('Processados....: ', teste.conta)
print('Normais........: ', teste.normais)
print('Anomalias......: ', teste.anomalias)

plt.rcParams["figure.figsize"] = (14, 14)
dados = dados.T
plt.grid()

# Plot amostras e centroides
dados = dados.T
if type_of_pre_pross == 'Autocloud':
    plt.plot(dados[0], dados[1], '.g')
    # Centroid
    # plt.plot(c_a1[0], c_a1[1], 'or')
else:
    plt.plot(dados, '.g')

# Plot AutoCloud centroids
for i in range(0, np.size(teste.c)):
    if type_of_pre_pross == 'Autocloud':
        plt.plot(teste.c[i].mean[0], teste.c[i].mean[1], 'X', color='orange')
    else:
        plt.plot(teste.c[i].mean, 'X', color='orange')

if not type_of_pre_pross == 'Autocloud':
    for k in teste.relacao_caso_status:
        if teste.relacao_caso_status[k][0]:
            plt.plot(teste.relacao_caso_status[k][1], '.r')
else:
    for k in teste.relacao_caso_status:
        if teste.relacao_caso_status[k][0]:
            plt.plot(teste.relacao_caso_status[k][1], '.r')

plt.legend(['Amostras', 'Auto-Cloud', 'Anomalias'])
plt.show()

elapsed_time = time.time() - start_time
mem = float(psutil.Process(os.getpid()).memory_full_info().uss) / 1048576
print(f'Elapsed Time: {elapsed_time} seconds')
print(f'Memory Used: {mem} MBs')
