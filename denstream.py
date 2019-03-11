import math
import numpy as np
from collections import deque, namedtuple
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import subprocess
from math import log10, sqrt
from Graph import createGraph, normGraph
from io import StringIO
import shutil

"""
A set of methods to control and plot DenStream data.
"""

def gen_data_plot(denstream, window_cases, alpha_range=(0, 1.0)):
    """
    This function organizes the necessary parameters for plotting.
    """
    alphas = np.linspace(*alpha_range, num=50)
    if len(window_cases) > 50:
        alphas = ([0] * (len(window_cases) - 50)) + list(alphas)
    points, outliers = [], []
    for i, (alpha, case) in enumerate(zip(alphas[-len(window_cases):],
                                      window_cases)):
        if denstream.is_normal(case.point):
            points.append(AlphaPoint(i=i, alpha=alpha, point=case.point))
        else:
            outliers.append(AlphaPoint(i=i, alpha=alpha, point=case.point))

    c_clusters, p_clusters = denstream.generate_clusters()
    o_clusters = denstream.generate_outlier_clusters()
    return points, outliers, c_clusters, p_clusters, o_clusters


def plot_clusters(process_name, total_cases, event_index, cp, points, outliers,
                  c_clusters, p_clusters, o_clusters, n, epsilon,
                  th, plot_path, cases_dict=None):
    """
    Plot all types of clusters, their respective graphs, and anomalous cases.
    Saves the plot according to the plot_path attribute.
    """
    plt.figure(figsize=(9, 7), dpi=400)
    ax = plt.subplot(212)
    plt.title(f'{process_name}, event index:{event_index}, total cases: {total_cases}, CP: {cp}')

    ax.set_xlim([-3, 3])
    ax.set_ylim([-1.5, 1.5])

    out = []
    if p_clusters[0]:
        for i, group in enumerate(p_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                ax.add_patch(patches.Circle(cluster.centroid,
                             cluster.radius, fill=False,
                             color='blue', ls=':'))
                ax.add_patch(patches.Circle(cluster.centroid,
                             epsilon, fill=False,
                             color='blue', ls='--'))
                ax.annotate(f'{i}', xy=cluster.centroid, color='red')
                # ax.annotate(f'{cluster.id}', xy=(cluster.centroid[0] + 0.05, cluster.centroid[1] + 0.05), color='green')

    if o_clusters:
        for cluster in o_clusters:
            if cluster.radius < 0.05:
                cluster.radius = 0.05
            ax.add_patch(patches.Circle(cluster.centroid,
                         cluster.radius, fill=False, color='red', ls='--'))
            ax.add_patch(patches.Circle(cluster.centroid,
                         epsilon, fill=False, color='red', ls='--'))
            # ax.annotate(f'{cluster.id}', xy=(cluster.centroid[0] + 0.05, cluster.centroid[1] + 0.05), color='green')


    if c_clusters[0]:
        for i, group in enumerate(c_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                ax.add_patch(patches.Circle(cluster.centroid,
                             cluster.radius, fill=False, color='black'))
                ax.add_patch(patches.Circle(cluster.centroid,
                             epsilon, fill=False, color='black', ls='--'))
                ax.annotate(f'{i}', xy=cluster.centroid, color='red')
                # ax.annotate(f'{cluster.id}', xy=(cluster.centroid[0] + 0.05, cluster.centroid[1] + 0.05), color='green')

    for p in points:
        ax.scatter(*p.point, alpha=p.alpha, color='black', marker='o', s=11)

    for p in outliers:
        ax.scatter(*p.point, alpha=p.alpha, color='orange', marker='x', s=11)

    # graphs
    count = 0
    total = 0

    if c_clusters[0] and cases_dict:
        for group in c_clusters:
            case_ids = []
            for cluster in group:
                case_ids.extend(cluster.case_ids)
            for id_ in case_ids:
                if id_ in cases_dict:
                    total += 1
                    break

    if p_clusters[0] and cases_dict:
        for group in p_clusters:
            case_ids = []
            for cluster in group:
                case_ids.extend(cluster.case_ids)
            for id_ in case_ids:
                if id_ in cases_dict:
                    total += 1
                    break

    graphs = deque()
    if c_clusters[0] and cases_dict:
        for i, group in enumerate(c_clusters):
            case_ids = []
            for cluster in group:
                case_ids.extend(cluster.case_ids)
            traces, times = [], []
            total_cases = 0
            cases_list_inside = []
            for id_ in case_ids:
                if id_ in cases_dict:
                    traces.append(cases_dict[id_]._trace)
                    times.append(cases_dict[id_]._timestamp)
                    cases_list_inside.append(id_)
                    total_cases += 1
            if total_cases:
                count += 1
                graph = createGraph(traces, times)
                if len(graph) == 0:
                    continue
                graph = normGraph(graph)
                unique_transitions = set()
                dot_buff = StringIO()
                dot_buff.write('graph {\ngraph [dpi = 400];\n')
                for (a, b), transition in graph.items():
                    if tuple(sorted((a, b))) not in unique_transitions:
                        weight = transition._count
                        if transition._time_norm == 0:
                            time = 0
                        else:
                            time = round(log10(transition._time_norm), 2)
                        dot_buff.write(f'{a} -- {b}[label="{weight} ({time})"];\n')
                        unique_transitions.add(tuple(sorted((a, b))))
                dot_buff.write('}')
                title = f'Core {i} - Total Cases {total_cases}'
                graphs.append((title, dot_buff, count, total, total_cases, i))


    # graphs
    if p_clusters[0] and cases_dict:
        for i, group in enumerate(p_clusters):
            case_ids = []
            for cluster in group:
                case_ids.extend(cluster.case_ids)
            traces, times = [], []
            total_cases = 0
            cases_list_inside = []
            for id_ in case_ids:
                if id_ in cases_dict:
                    traces.append(cases_dict[id_]._trace)
                    times.append(cases_dict[id_]._timestamp)
                    cases_list_inside.append(id_)
                    total_cases += 1
            if total_cases:
                count += 1
                graph = createGraph(traces, times)
                if len(graph) == 0:
                    continue
                graph = normGraph(graph)
                unique_transitions = set()

                dot_buff = StringIO()
                dot_buff.write('graph {\ngraph [dpi = 400];\n')
                for (a, b), transition in graph.items():
                    if tuple(sorted((a, b))) not in unique_transitions:
                        weight = transition._count
                        if transition._time_norm == 0:
                            time = 0
                        else:
                            time = round(log10(transition._time_norm), 2)
                        dot_buff.write(f'{a} -- {b}[label="{weight} ({time})"];\n')
                        unique_transitions.add(tuple(sorted((a, b))))
                dot_buff.write('}')
                title = f'P-micro {i} - Total Cases {total_cases}'
                graphs.append((title, dot_buff, count, total, total_cases, i))

    if graphs:
        for graph in graphs:
            gen_graphviz(graph)
        for title, dot_buf, count, total, total_cases, i in graphs:
            ax = plt.subplot(2, total, count)
            ax.imshow(mpimg.imread(f'aux/graphvizfiles/{count}.png'))
            ax.set_title(f'{title}')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'{plot_path}/plot_{n}.png')
    plt.close()

def cluster_metrics(total_cases, event_index, cp,
                    c_clusters, p_clusters, o_clusters):
    """
    Generates cluster metrics to record them.
    Works in conjunction with Process.clusterMetrics().
    """
    out = []
    if p_clusters[0]:
        for i, group in enumerate(p_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                out.append([event_index, cluster.centroid[0],\
                            cluster.centroid[1], cluster.radius,\
                            cluster.weight, total_cases,\
                            cp, 'p-micro', cluster.id])

    if o_clusters:
        for cluster in o_clusters:
            if cluster.radius < 0.05:
                cluster.radius = 0.05
            out.append([event_index, cluster.centroid[0],\
                        cluster.centroid[1], cluster.radius,\
                        cluster.weight, total_cases,\
                        cp, 'o-micro', cluster.id])

    if c_clusters[0]:
        for i, group in enumerate(c_clusters):
            for cluster in group:
                if cluster.radius < 0.05:
                    cluster.radius = 0.05
                out.append([event_index, cluster.centroid[0],\
                            cluster.centroid[1], cluster.radius,\
                            cluster.weight, total_cases,\
                            cp, 'c-micro', cluster.id])

    return out

def gen_graphviz(p):
    """
    Generates the graphs of each cluster using graphviz technology
    and saves them in an auxiliary folder, which is later retrieved
    by plot_clusters() for the complete plotting.
    """
    title, dot_buf, count, total, total_cases, i = p
    with open(f'aux/dotfiles/{count}.dot', 'w') as fd:
        dot_buf.seek(0)
        shutil.copyfileobj(dot_buf, fd)

    cmd = f'dot -Tpng aux/dotfiles/{count}.dot -o aux/graphvizfiles/{count}.png'
    subprocess.Popen(cmd, shell=True).wait()

AlphaPoint = namedtuple('AlphaPoint', ['i', 'alpha', 'point'])
Case = namedtuple('Case', ['id', 'point'])

class DenStream:
    """
    Manages the DenStream algorithm and implements
    classes MicroCluster and Cluster.
    """
    def __init__(self, n_features, lambda_, beta, epsilon, mu, stream_speed):
        """
        Initializes the DenStream class.
        """
        self._n_features = n_features
        self._lambda = lambda_
        self._beta = beta
        self._epsilon = epsilon
        self._mu = mu
        self._p_micro_clusters = {}
        self._o_micro_clusters = {}
        self._label = 0
        self._time = 0
        self._initiated = False
        self._all_cases = set()
        self._stream_speed = stream_speed

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
            # decay all p_micro_clusters weights except for the cluster i
            # self.decay_p_mc(i)
            # self.decay_o_mc()
        else:
            # decay all p_micro_clusters weights
            # self.decay_p_mc()
            i, closest_o_mc, _ = self.find_closest_o_mc(case.point)
            # Try to merge point with closest o_mc
            if (closest_o_mc and
               closest_o_mc.radius_with_new_point(case.point) <= self._epsilon):
                closest_o_mc.update(case)
                # decay all o_micro_clusters weights except for the cluster i
                self.decay_o_mc(i)
                # Try to promote o_micro_clusters to p_mc
                if closest_o_mc._weight > self._beta * self._mu:
                    del self._o_micro_clusters[i]
                    self._p_micro_clusters[self._label] = closest_o_mc
            else:
                # decay all o_mc weights
                self.decay_o_mc()
                # create new o_mc containing the new point
                new_o_mc = self.MicroCluster(n_features=self._n_features,
                                             creation_time=t,
                                             lambda_=self._lambda,
                                             stream_speed=self._stream_speed)
                new_o_mc.update(case)
                self._label += 1
                self._o_micro_clusters[self._label] = new_o_mc

        for i, cluster in chain(self._p_micro_clusters.items(), self._o_micro_clusters.items()):
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
