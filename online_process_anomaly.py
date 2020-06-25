# -*- coding: utf-8 -*-
"""online_anomaly_detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t-SwzQ9VoX80bNqQPhXpvndI-jX63lgq

### Clona repositórito do Github
nota: adicionar login e usuário porque o repositório é privado
"""

import os
from getpass import getpass

import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time


class Case:
    activities = []
    label = ''

    def __init__(self, id):
        self.id = str(id)

    def __repr__(self):
        return str(self.id)

    def updateTrace(self, activity_name):
        self.activities.append(activity_name)

    def updateLabel(self, label):
        self.label = label


class DataCloud:
    N = 0

    def __init__(self, x):
        self.n = 1
        self.mean = x
        self.variance = 0
        self.pertinency = 1
        DataCloud.N += 1

    def addDataClaud(self, x):
        self.n = 2
        self.mean = (self.mean + x) / 2
        self.variance = ((np.linalg.norm(self.mean - x)) ** 2)

    def updateDataCloud(self, n, mean, variance):
        self.n = n
        self.mean = mean
        self.variance = variance


class AutoCloud:
    c = np.array([DataCloud(0)], dtype=DataCloud)
    alfa = np.array([0.0], dtype=float)
    intersection = np.zeros((1, 1), dtype=int)
    listIntersection = np.zeros((1), dtype=int)
    matrixIntersection = np.zeros((1, 1), dtype=int)
    relevanceList = np.zeros((1), dtype=int)
    k = 1
    relacao_caso_status = {}

    def __init__(self, m):
        AutoCloud.m = m
        AutoCloud.c = np.array([DataCloud(0)], dtype=DataCloud)
        AutoCloud.alfa = np.array([0.0], dtype=float)
        AutoCloud.intersection = np.zeros((1, 1), dtype=int)
        AutoCloud.listIntersection = np.zeros((1), dtype=int)
        AutoCloud.relevanceList = np.zeros((1), dtype=int)
        AutoCloud.matrixIntersection = np.zeros((1, 1), dtype=int)
        AutoCloud.k = 1
        AutoCloud.classIndex = [[1], [1]]

    def plotgrafico(self, x, y, index):
        for data in y:
            plt.plot(x, data, '.', color=listaCor[index])

    def mergeClouds(self):
        i = 0
        while (i < len(AutoCloud.listIntersection) - 1):
            merge = False
            j = i + 1
            while (j < len(AutoCloud.listIntersection)):
                # print("i",i,"j",j,"l",np.size(AutoCloud.listIntersection),"m",np.size(AutoCloud.matrixIntersection),"c",np.size(AutoCloud.c))
                if (AutoCloud.listIntersection[i] == 1 and AutoCloud.listIntersection[j] == 1):
                    AutoCloud.matrixIntersection[i, j] = AutoCloud.matrixIntersection[i, j] + 1;
                nI = AutoCloud.c[i].n
                nJ = AutoCloud.c[j].n
                meanI = AutoCloud.c[i].mean
                meanJ = AutoCloud.c[j].mean
                varianceI = AutoCloud.c[i].variance
                varianceJ = AutoCloud.c[j].variance
                nIntersc = AutoCloud.matrixIntersection[i, j]
                if (nIntersc > (nI - nIntersc) or nIntersc > (nJ - nIntersc)):
                    merge = True
                    # update values
                    n = nI + nJ - nIntersc
                    mean = ((nI * meanI) + (nJ * meanJ)) / (nI + nJ)
                    variance = ((nI - 1) * varianceI + (nJ - 1) * varianceJ) / (nI + nJ - 2)
                    newCloud = DataCloud(mean)
                    newCloud.updateDataCloud(n, mean, variance)
                    # atualizando lista de interseção
                    AutoCloud.listIntersection = np.concatenate((AutoCloud.listIntersection[0: i], np.array([1]),
                                                                 AutoCloud.listIntersection[i + 1: j],
                                                                 AutoCloud.listIntersection[
                                                                 j + 1: np.size(AutoCloud.listIntersection)]),
                                                                axis=None)
                    # atualizando lista de data clouds
                    AutoCloud.c = np.concatenate((AutoCloud.c[0: i], np.array([newCloud]), AutoCloud.c[i + 1: j],
                                                  AutoCloud.c[j + 1: np.size(AutoCloud.c)]), axis=None)
                    # update  intersection matrix
                    M0 = AutoCloud.matrixIntersection
                    # Remover linhas
                    M1 = np.concatenate((M0[0: i, :], np.zeros((1, len(M0))), M0[i + 1: j, :], M0[j + 1: len(M0), :]))
                    # remover colunas
                    M1 = np.concatenate((M1[:, 0: i], np.zeros((len(M1), 1)), M1[:, i + 1: j], M1[:, j + 1: len(M0)]),
                                        axis=1)
                    # calculando nova coluna
                    col = (M0[:, i] + M0[:, j]) * (M0[:, i] * M0[:, j] != 0)
                    col = np.concatenate((col[0: j], col[j + 1: np.size(col)]))
                    # calculando nova linha
                    lin = (M0[i, :] + M0[j, :]) * (M0[i, :] * M0[j, :] != 0)
                    lin = np.concatenate((lin[0: j], lin[j + 1: np.size(lin)]))
                    # atualizando coluna
                    M1[:, i] = col
                    # atualizando linha
                    M1[i, :] = lin
                    M1[i, i + 1: j] = M0[i, i + 1: j] + M0[i + 1: j, j].T;
                    AutoCloud.matrixIntersection = M1
                j += 1
            if (merge):
                i = 0
            else:
                i += 1

    def run(self, X, case_id):
        AutoCloud.listIntersection = np.zeros((np.size(AutoCloud.c)), dtype=int)
        if AutoCloud.k == 1:
            AutoCloud.c[0] = DataCloud(X)

        elif AutoCloud.k == 2:
            AutoCloud.c[0].addDataClaud(X)
        elif AutoCloud.k >= 3:
            i = 0
            createCloud = True
            AutoCloud.alfa = np.zeros((np.size(AutoCloud.c)), dtype=float)
            for data in AutoCloud.c:
                n = data.n + 1
                mean = ((n - 1) / n) * data.mean + (1 / n) * X
                variance = ((n - 1) / n) * data.variance + (1 / n) * ((np.linalg.norm(X - mean)) ** 2)
                eccentricity = (1 / n) + ((mean - X).T.dot(mean - X)) / (n * variance)
                typicality = 1 - eccentricity
                norm_eccentricity = eccentricity / 2
                norm_typicality = typicality / (AutoCloud.k - 2)

                if (norm_eccentricity > (AutoCloud.m ** 2 + 1) / (2 * n)):
                    self.relacao_caso_status[case_id] = [case_id, norm_eccentricity > (AutoCloud.m ** 2 + 1) / (2 * n),
                                                         mean]

                if (norm_eccentricity <= (AutoCloud.m ** 2 + 1) / (2 * n)):
                    data.updateDataCloud(n, mean, variance)
                    AutoCloud.alfa[i] = norm_typicality
                    createCloud = False
                    AutoCloud.listIntersection.itemset(i, 1)
                else:
                    AutoCloud.alfa[i] = norm_typicality
                    AutoCloud.listIntersection.itemset(i, 0)
                i += 1

            if (createCloud):
                AutoCloud.c = np.append(AutoCloud.c, DataCloud(X))
                AutoCloud.listIntersection = np.insert(AutoCloud.listIntersection, i, 1)
                AutoCloud.matrixIntersection = np.pad(AutoCloud.matrixIntersection, ((0, 1), (0, 1)), 'constant',
                                                      constant_values=(0))
            self.mergeClouds()
            AutoCloud.relevanceList = AutoCloud.alfa / np.sum(AutoCloud.alfa)
            AutoCloud.classIndex.append(np.argmax(AutoCloud.relevanceList))
            AutoCloud.classIndex.append(AutoCloud.alfa)

        AutoCloud.k = AutoCloud.k + 1


class DenStream:
    """
    Manages the DenStream algorithm and implements
    classes MicroCluster and Cluster.
    """

    def __init__(self, n_features, outlier_threshold, decay_factor, epsilon, mu, stream_speed, ncluster):
        """
        Initializes the DenStream class.
        """
        self._n_features = n_features
        self.outlier_threshold = outlier_threshold
        self.decay_factor = decay_factor
        self._epsilon = epsilon
        self._mu = mu
        self._p_micro_clusters = {}
        self._o_micro_clusters = {}
        self._label = 0
        self._time = 0
        self._initiated = False
        self._all_cases = set()
        self._stream_speed = stream_speed
        self._ncluster = ncluster

    @staticmethod
    def euclidean_distance(point1, point2):
        """
        Compute the Euclidean Distance between two points.
        """
        return np.sqrt(np.sum(np.power(point1 - point2, 2)))

    def find_closest_p_mc(self, point):
        """
        Find the closest p_micro_cluster to the point "point" according
        to the Euclidean Distance between it and the cluster's centroid.
        """
        if len(self._p_micro_clusters) == 0:
            return None, None, None
        distances = [(i, self.euclidean_distance(point, cluster.centroid))
                     for i, cluster in self._p_micro_clusters.items()]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        return i, self._p_micro_clusters[i], dist

    def find_closest_o_mc(self, point):
        """
        Find the closest o_micro_cluster to the point "point" according
        to the Euclidean Distance between it and the cluster's centroid.
        """
        if len(self._o_micro_clusters) == 0:
            return None, None, None
        distances = [(i, self.euclidean_distance(point, cluster.centroid))
                     for i, cluster in self._o_micro_clusters.items()]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        return i, self._o_micro_clusters[i], dist

    def decay_p_mc(self, last_mc_updated_index=None):
        """
        Decay the weight of all p_micro_clusters for the exception
        of an optional parameter last_mc_updated_index
        """
        for i, cluster in self._p_micro_clusters.items():
            if i != last_mc_updated_index:
                cluster.update(None)

    def decay_o_mc(self, last_mc_updated_index=None):
        """
        Decay the weight of all o_micro_clusters for the exception
        of an optional parameter last_mc_updated_index
        """
        for i, cluster in self._o_micro_clusters.items():
            if i != last_mc_updated_index:
                cluster.update(None)

    def merge(self, case, t):
        """
        Try to add a point "point" to the existing p_micro_clusters at time "t"
        Otherwise, try to add that point to the existing o_micro_clusters
        If all fails, create a new o_micro_cluster with that new point
        """
        i, closest_p_mc, _ = self.find_closest_p_mc(case.point)
        # Try to merge point with closest p_mc
        if (closest_p_mc and
                closest_p_mc.radius_with_new_point(case.point) <= self._epsilon):
            closest_p_mc.update(case)
        else:
            i, closest_o_mc, _ = self.find_closest_o_mc(case.point)
            # Try to merge point with closest o_mc
            if (closest_o_mc and
                    closest_o_mc.radius_with_new_point(case.point) <= self._epsilon):
                closest_o_mc.update(case)
                # Try to promote o_micro_clusters to p_mc
                if closest_o_mc._weight > self._beta * self._mu:
                    del self._o_micro_clusters[i]
                    self._p_micro_clusters[self._label] = closest_o_mc
            else:
                # create new o_mc containing the new point
                new_o_mc = self.MicroCluster(n_features=self._n_features,
                                             creation_time=t,
                                             lambda_=self._lambda,
                                             stream_speed=self._stream_speed)
                new_o_mc.update(case)
                self._label += 1
                self._o_micro_clusters[self._label] = new_o_mc

        for i, cluster in self._p_micro_clusters.items():
            if cluster._count_to_decay == 0:
                cluster.update(None)
                cluster._count_to_decay = cluster._stream_speed
                if cluster in self._p_micro_clusters and cluster._weight < self._beta * self._mu:
                    del self._p_micro_clusters[i]
                    self._o_micro_clusters[i] = cluster
            else:
                cluster._count_to_decay = cluster._count_to_decay - 1
        for i, cluster in self._o_micro_clusters.items():
            if cluster._count_to_decay == 0:
                cluster.update(None)
                cluster._count_to_decay = cluster._stream_speed
            else:
                cluster._count_to_decay = cluster._count_to_decay - 1

    def train(self, case):
        """
        "Train" Denstream by updating its p_micro_clusters and o_micro_clusters
        with a new point "point"
        """
        # clean case
        if case.id in self._all_cases:
            removed = False
            for mc in self._p_micro_clusters.values():
                if case.id in mc._case_ids:
                    mc._case_ids.remove(case.id)
                    removed = True
                    break
            if not removed:
                for mc in self._o_micro_clusters.values():
                    if case.id in mc._case_ids:
                        mc._case_ids.remove(case.id)
                        break
        else:
            self._all_cases.add(case.id)

        self._time += 1
        if not self._initiated:
            raise Exception

        t = self._time
        # Compute Tp
        try:
            part = (self._beta * self._mu) / (self._beta * self._mu - 1)
            Tp = math.ceil(1 / self._lambda * math.log2(part))
        except:
            Tp = 1

        # Add point
        self.merge(case, self._time)

        # Test if should remove any p_micro_cluster or o_micro_cluster
        if t % Tp == 0:
            for i in list(self._p_micro_clusters.keys()):
                cluster = self._p_micro_clusters[i]
                if cluster._weight < self._beta * self._mu:
                    del self._p_micro_clusters[i]

            for i in list(self._o_micro_clusters.keys()):
                cluster = self._o_micro_clusters[i]
                to = cluster._creation_time
                e = ((math.pow(2, - self._lambda * (t - to + Tp)) - 1) /
                     (math.pow(2, - self._lambda * Tp) - 1))
                if cluster._weight < e:
                    del self._o_micro_clusters[i]

    def is_normal(self, point):
        """
        Find if point "point" is inside any p_micro_cluster
        """
        if len(self._p_micro_clusters) == 0:
            return False

        distances = [(i, self.euclidean_distance(point, cluster.centroid))
                     for i, cluster in self._p_micro_clusters.items()]
        for i, dist in distances:
            if dist <= self._epsilon:
                return True
        return False

    def DBSCAN(self, buffer):
        """
        Perform DBSCAN to create initial p_micro_clusters
        Works by grouping points with distance <= self._epsilon
        and filtering groups that are not dense enough (weight >= beta * mu)
        """
        used_cases = set()
        for case in (case for case in buffer if case.id not in used_cases):
            used_cases.add(case.id)
            group = [case]
            for other_case in (case for case in buffer if case.id not in used_cases):
                dist = self.euclidean_distance(case.point,
                                               other_case.point)
                if dist <= self._epsilon:
                    group.append(other_case)

            weight = len(group)
            if weight >= self._beta * self._mu:
                new_p_mc = self.MicroCluster(n_features=self._n_features,
                                             creation_time=0,
                                             lambda_=self._lambda,
                                             stream_speed=self._stream_speed)
                for case in group:
                    used_cases.add(case.id)
                    new_p_mc.update(case)
                    self._all_cases.add(case.id)
                self._label += 1
                self._p_micro_clusters[self._label] = new_p_mc

            else:
                used_cases.remove(case.id)
        self._initiated = True

    def generate_clusters(self):
        """
        Perform DBSCAN to create the final c_micro_clusters
        Works by grouping dense enough p_micro_clusters (weight >= mu)
        with distance <= 2 * self._epsilon
        """
        if len(self._p_micro_clusters) > 1:
            connected_clusters = []
            remaining_clusters = deque((self.Cluster(id=i,
                                                     centroid=mc.centroid,
                                                     radius=mc.radius,
                                                     weight=mc._weight,
                                                     case_ids=mc._case_ids)
                                        for i, mc in self._p_micro_clusters.items()))

            testing_group = -1
            # try to add the remaining clusters to existing groups
            while remaining_clusters:
                # create a new group
                connected_clusters.append([remaining_clusters.popleft()])
                testing_group += 1
                change = True
                while change:
                    change = False
                    buffer_ = deque()
                    # try to add remaining clusters
                    # to the existing group as it is
                    # if we add a new cluster to that group,
                    # perform the check again
                    while remaining_clusters:
                        r_cluster = remaining_clusters.popleft()
                        to_add = False
                        for cluster in connected_clusters[testing_group]:
                            dist = self.euclidean_distance(cluster.centroid,
                                                           r_cluster.centroid)
                            if dist <= 2 * self._epsilon:
                                to_add = True
                                break
                        if to_add:
                            connected_clusters[testing_group].append(r_cluster)
                            change = True
                        else:
                            buffer_.append(r_cluster)
                    remaining_clusters = buffer_

            dense_groups, not_dense_groups = [], []
            for group in connected_clusters:
                if sum([c.weight for c in group]) >= self._mu:
                    dense_groups.append(group)
                else:
                    not_dense_groups.append(group)
            if len(dense_groups) == 0:
                dense_groups = [[]]
            if len(not_dense_groups) == 0:
                not_dense_groups = [[]]

            return dense_groups, not_dense_groups

        # only one p_micro_cluster (check if it is dense enough)
        elif len(self._p_micro_clusters) == 1:
            mc = list(self._p_micro_clusters.values())[0]
            id = list(self._p_micro_clusters.keys())[0]
            centroid = mc.centroid
            radius = mc.radius
            case_ids = mc._case_ids
            weight = mc._weight
            if weight >= self._mu:
                return [[self.Cluster(id=id,
                                      centroid=centroid,
                                      radius=radius,
                                      weight=weight,
                                      case_ids=case_ids)]], [[]]
            else:
                return [[]], [[self.Cluster(id=id,
                                            centroid=centroid,
                                            radius=radius,
                                            weight=weight,
                                            case_ids=case_ids)]]
        return [[]], [[]]

    def generate_outlier_clusters(self):
        """
        Generates a list of o-micro-clusters.
        """
        return [self.Cluster(id=i,
                             centroid=mc.centroid,
                             radius=mc.radius,
                             weight=mc._weight,
                             case_ids=mc._case_ids)
                for i, mc in self._o_micro_clusters.items()]

    class MicroCluster:
        """
        The class represents a micro-cluster and its attributes.
        """

        def __init__(self, n_features, creation_time, lambda_, stream_speed):
            """
            Initializes the MicroCluster attributes.
            """
            self._CF = np.zeros(n_features)
            self._CF2 = np.zeros(n_features)
            self._weight = 0
            self._creation_time = creation_time
            self._case_ids = set()
            self._lambda = lambda_
            self._stream_speed = stream_speed
            self._count_to_decay = self._stream_speed

        @property
        def centroid(self):
            """
            Computes and returns the micro-cluster's centroid value,
            which is given by CF divided by weight.
            """
            return self._CF / self._weight

        @property
        def radius(self):
            """
            Computes and returns the micro-cluster's radius.
            """
            A = np.sqrt(np.sum(np.square(self._CF2))) / self._weight
            B = np.square(np.sqrt(np.sum(np.square(self._CF))) / self._weight)
            S = A - B
            if S < 0:
                S = 0
            return sqrt(S)

        def radius_with_new_point(self, point):
            """
            Computes the micro-cluster's radius considering a new point.
            The returned value is then compared to self._epsilon to check
            whether the point must be absorbed or not.
            """
            CF1 = self._CF + point
            CF2 = self._CF2 + point * point
            weight = self._weight + 1

            A = np.sqrt(np.sum(np.square(CF2))) / weight
            B = np.square(np.sqrt(np.sum(np.square(CF1))) / weight)
            S = A - B
            if S < 0:
                S = 0
            return sqrt(S)

        def update(self, case):
            """
            Updates the micro-cluster weights either
            considering a new case or not.
            """
            if case is None:
                factor = 2 ** (-self._lambda)
                self._CF *= factor
                self._CF2 *= factor
                self._weight *= factor
            else:
                self._CF += case.point
                self._CF2 += case.point * case.point
                self._weight += 1
                self._case_ids.add(case.id)

    class Cluster:
        """
        Class that represents a cluster.
        """

        def __init__(self, id, centroid, radius, weight, case_ids):
            """
            Initializes a cluster.
            """
            self.id = id
            self.centroid = centroid
            self.radius = radius
            self.weight = weight
            self.case_ids = case_ids

        # def __str__(self):
        #     return f'{self.id} - Centroid: {self.centroid} | Radius: {self.radius}'


def read_log(path, log):
    '''
    Reads and preprocesses event log
    '''
    df_raw = pd.read_csv(f'{path}/{log}')
    df_raw['activity'] = df_raw['activity_name'].str.replace(' ', '-')
    df_proc = df_raw[['case_id', 'activity', 'label']]
    del df_raw

    return df_proc


def cases_y_list(df):
    '''
    Creates a list of cases (and their respective labels) for model training
    '''
    cases, case_id = [], []
    for group in df.groupby('case_id'):
        case = Case(group[0])
        case.activities = list(group[1].activity)
        case.label = list(group[1].label)[0]
        cases.append(case)
        case_id.append(group[0])

    return cases, case_id


def create_model(cases, dimensions, window, min_count):
    '''
    Creates a word2vec model
    '''
    print('Creates a word2vec model.')
    model = Word2Vec(
        size=dimensions,
        window=window,
        min_count=min_count,
        workers=-1)
    model.build_vocab(cases)
    model.train(cases, total_examples=len(cases), epochs=10)
    model.wv.vocab
    model.save("current_model")

    return model


def update_model(cases):
    '''
    TODO
    Updates word2vec model
    '''
    new_model = Word2Vec.load("current_model")
    new_model.build_vocab(cases, update=True)
    new_model.train(cases, total_examples=2, epochs=1)
    new_model.wv.vocab

    return new_model


def average_case_vector(trace, model):
    '''
    Computes the average case feature vector according to a model
    '''
    case_vector = []
    for token in trace:
        try:
            case_vector.append(model.wv[token])
        except KeyError:
            pass
    return np.array(case_vector).mean(axis=0)


def average_vectors(traces, model):
    '''
    Computes average feature vector for several cases
    '''
    vectors = []
    for trace in traces:
        vectors.append(average_case_vector(trace, model))

    return vectors


def clean_case_memory(cases_in_memory):
    '''
    TODO
    Deletes older cases
    '''
    cases = cases_in_memory
    return cases


"""### Inicialização da base"""

path = 'data/'
log = 'small-0.0-1.csv'

df = read_log(path, log)

"""### Criação do modelo word2vec
O número de eventos usados para criação do modelo é definido a partir da variável `stream_window` (tamanho da janela). Com isso, os eventos são agrupados em cases e o modelo é inicializado. Aqui define-se também os parâmetros do word2vec.
"""

stream_window = 1000
stream_windowed = 0
dimensions_word2vec = 50
window_word2vec = 3
minimum_word2vec = 3

case_ids = []
kkk = pd.DataFrame([])

df_train = df.iloc[:stream_window]
cases_in_memory, case_ids = cases_y_list(df_train)

train_word2vec = []
for case in cases_in_memory:
    train_word2vec.append(case.activities)

word2vec_model = create_model(train_word2vec, dimensions_word2vec, window_word2vec, minimum_word2vec)

"""### Inicialização do AutoCloud
Calcula-se os vetores médios dos cases utilizados para criação do modelo word2vec. Após isso, o AutoCloud é inicializado e alimentado com os vetores iniciais.
"""

m = 1.7
auto_cloud = AutoCloud(m)
vector = np.ndarray([])
vector_n = []

vectors = average_vectors(train_word2vec, word2vec_model)
for vector in vectors:
    auto_cloud.run(vector, 0)

"""### Processamento da stream de eventos
Aqui iteramos pelo dataframe que contém os eventos (simulando uma stream). Para cada evento, verificamos se seu case já existe na lista de cases e atualizamos a lista.

**TODO**:
* Atualizar lista de cases a partir do janelamento
* Atualizar modelo word2vec (treinar com os cases mais recentes)
* Calcular métricas da clusterização
"""

stream_windowed = stream_window

for event in df.iloc[stream_window:].values:

    if next((case for case in cases_in_memory if case.id == str(event[0])), None):
        '''
        Case já existe. Portanto, atualizamos o valor dele na lista de cases
        '''
        case.updateTrace(event[1])
    else:
        '''
        Case não existe. Portanto, criamos um novo
        '''
        case = Case(event[0])
        case.activities = list(event[1])
        case.label = event[2]
        cases_in_memory.append(case)
        case_ids.append(case)

    vector = average_case_vector(case.activities, word2vec_model)

    # vector_n.append(vector)

    auto_cloud.run(vector, event[0])

    # TODO tem que retreinar apenas os ultimos da janela para isso tem que atualizar os cases
    if np.size(case_ids) == stream_windowed:
        stream_windowed += stream_window
        train_word2vec = []
        for case in cases_in_memory:
            word2vec_model = update_model(case.activities)

    # if condicao_janela (stream_window):
    #     cases_in_memory = clean_case_memory(cases_in_memory)
    #     word2vec_model = update_model(selected_cases, word2vec_model)

print(auto_cloud.alfa)
print(vector)
print(np.size(auto_cloud.c))
print(vector.T)

for i in range(0, np.size(vector)):
    plt.plot(vector, '.g')

# Plot AutoCloud centroids
for i in range(0, np.size(auto_cloud.c)):
    plt.plot(auto_cloud.c[i].mean[0], auto_cloud.c[i].mean[1], 'X', color='orange')

# Plot AutoCloud outliers
print('Teste: ', auto_cloud.relacao_caso_status)
for i in range(1, np.size(auto_cloud.relacao_caso_status)):
    plt.plot(auto_cloud.relacao_caso_status[i][3][0], auto_cloud.relacao_caso_status[i][3][1], '*', color='red')

plt.show()

# hyperparameters configuration :: DenStream
n_features = 2
decay_factor_ = 0.15
outlier_threshold_ = 0.3
epsilon = 0.1
mu = 4
stream_speed = 1000

start_time = time.time()
event_stream = pd.read_csv(f'{path}/{log}')
event_stream = event_stream.values

denstream_kwargs = {'n_features': n_features,
                    'outlier_threshold': outlier_threshold_,
                    'decay_factor': decay_factor_,
                    'epsilon': epsilon,
                    'mu': mu,
                    'stream_speed': stream_speed,
                    'ncluster': 0}

_denstream = DenStream(**denstream_kwargs)

# initialise denstream
_denstream.DBSCAN(vector)

print(_denstream)