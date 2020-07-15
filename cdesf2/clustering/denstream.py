import math
import numpy as np
from collections import deque
from typing import Tuple, List, Union
from ..data_structures import MicroCluster
from ..data_structures import Cluster
from ..data_structures import Case


class DenStream:
    """
    Manages the DenStream algorithm.
    """
    def __init__(self, n_features: int, lambda_: float, beta: float,
                 epsilon: float, mu: int, stream_speed: int, ncluster: int):
        """
        Initializes the DenStream class.

        Parameters
        --------------------------------------
        n_features: int
            The number of features DenStream must consider,
            in our case is always set to 2, since we have
            two attributes (graph_distance and time_distance)
        lambda_: float
            Sets the importance of historical data for the
            current clusters
        beta: float
            Defines the threshold for a micro-cluster weight
            for that instance to be considered inside the cluster
        epsilon: float
            Defines the maximum distance between an instance
            and a cluster for that instance to be considered
            inside the cluster
        mu: int
            It is the maximum weight an overall neighbourhood
            needs to be considered a core object
        stream_speed: int
        ncluster: int
        """
        self.n_features = n_features
        self.lambda_ = lambda_
        self.beta = beta
        self.epsilon = epsilon
        self.mu = mu
        self.p_micro_clusters = {}
        self.o_micro_clusters = {}
        self.label = 0
        self.time = 0
        self.initiated = False
        self.all_cases = set()
        self.stream_speed = stream_speed
        # don't know if ncluster is necessary: never used in functions
        self.ncluster = ncluster

    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> np.float64:
        """
        Compute the Euclidean Distance between two points.

        Parameters
        --------------------------------------
        point1: np.ndarray,
            Array position of a point
        point2: np.ndarray,
            Array position of a point
        Returns
        --------------------------------------
        The Euclideand Distance: np.float64
        """
        return np.sqrt(np.sum(np.power(point1 - point2, 2)))

    def find_closest_mc(self, point: np.ndarray, micro_cluster_dict: dict) \
            -> Union[Tuple[int, MicroCluster, np.float64], Tuple[None, None, None]]:
        """
        Find the closest p_micro_cluster or the closest o_micro_cluster
        to the point "point" according to the Euclidean Distance
        between it and the cluster's centroid.

        Parameters
        --------------------------------------
        point: np.ndarray,
            Array position of a point
        micro_cluster_dict: dict,
            The dictionary of micro-clusters, it
            can be p_micro_clusters or o_micro_cluster
            depending if you want to find the closest
            p-micro-cluster or the closest o-micro-cluster
        Returns
        --------------------------------------
        i: int,
            Index of the micro-cluster
        micro_cluster_dict: MicroCluster,
            The correspondent p-mciro-cluster
            or o-micro-cluster which the point was added
        dist: np.float64,
            Distance between the micro-cluster and the point
        """
        if len(micro_cluster_dict) == 0:
            return None, None, None
        distances = [(i, self.euclidean_distance(point, cluster.centroid))
                     for i, cluster in micro_cluster_dict.items()]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        return i, micro_cluster_dict[i], dist

    #
    # def decay_p_mc(self, last_mc_updated_index=None):
    #    """
    #    Decay the weight of all p_micro_clusters for the exception
    #    of an optional parameter last_mc_updated_index
    #
    #   Parameters
    #    --------------------------------------
    #    last_mc_updated_index: int
    #    """
    #    for i, cluster in self.p_micro_clusters.items():
    #        if i != last_mc_updated_index:
    #            cluster.update(None)

    # def decay_o_mc(self, last_mc_updated_index=None):
    #    """
    #    Decay the weight of all o_micro_clusters for the exception
    #    of an optional parameter last_mc_updated_index
    #
    #    Parameters
    #    --------------------------------------
    #    last_mc_updated_index: int
    #    """
    #    for i, cluster in self.o_micro_clusters.items():
    #        if i != last_mc_updated_index:
    #            cluster.update(None)

    def merge(self, case: Case, t: int) -> None:
        """
        Try to add a point to the existing p_micro_clusters at time "t"
        Otherwise, try to add that point to the existing o_micro_clusters
        If all fails, create a new o_micro_cluster with that new point

        Parameters
        --------------------------------------
        case: Case
            A Case object, with all its attributes
        t: int
            Time in single units
        """
        case_point = np.array([case.graph_distance, case.time_distance])
        i, closest_p_mc, _ = self.find_closest_mc(case_point, self.p_micro_clusters)
        # Try to merge point with closest p_mc
        if (closest_p_mc and
           closest_p_mc.radius_with_new_point(case_point) <= self.epsilon):
            closest_p_mc.update(case)
        else:
            i, closest_o_mc, _ = self.find_closest_mc(case_point, self.o_micro_clusters)
            # Try to merge point with closest o_mc
            if (closest_o_mc and
               closest_o_mc.radius_with_new_point(case_point) <= self.epsilon):
                closest_o_mc.update(case)
                # Try to promote o_micro_clusters to p_mc
                if closest_o_mc.weight > self.beta * self.mu:
                    del self.o_micro_clusters[i]
                    self.p_micro_clusters[self.label] = closest_o_mc
            else:
                # create new o_mc containing the new point
                new_o_mc = MicroCluster(n_features=self.n_features,
                                        creation_time=t,
                                        lambda_=self.lambda_,
                                        stream_speed=self.stream_speed)
                new_o_mc.update(case)
                self.label += 1
                self.o_micro_clusters[self.label] = new_o_mc

        self.decay_p_micro_cluster()

    def decay_p_micro_cluster(self) -> None:
        """
        Decays p-micro-clusters and makes the micro-cluster with
        count_to_decay = 0 an o-micro-cluster
        """
        for i, cluster in self.p_micro_clusters.items():
            # if the p-micro-cluster has the counter at 0,
            # it will be deleted from the dictionary of p-micro-clusters
            # and it will be pushed in the dictionary of o-micro-clusters
            if cluster.count_to_decay == 0:
                cluster.update(None)
                cluster.count_to_decay = cluster.stream_speed
                if cluster in self.p_micro_clusters and cluster.weight < self.beta * self.mu:
                    del self.p_micro_clusters[i]
                    self.o_micro_clusters[i] = cluster
            # if the counter is not at 0, it will be decremented by 1
            else:
                cluster.count_to_decay = cluster.count_to_decay - 1
        for i, cluster in self.o_micro_clusters.items():
            if cluster.count_to_decay == 0:
                cluster.update(None)
                cluster.count_to_decay = cluster.stream_speed
            else:
                cluster.count_to_decay = cluster.count_to_decay - 1

    def train(self, case: Case):
        """
        "Train" Denstream by updating its p_micro_clusters and o_micro_clusters
        with a new point

        Parameters
        --------------------------------------
        case: Case
            A Case object, with all its attributes
        """
        # clean case
        if case.id in self.all_cases:
            removed = False
            for mc in self.p_micro_clusters.values():
                if case.id in mc.case_ids:
                    mc.case_ids.remove(case.id)
                    removed = True
                    break
            if not removed:
                for mc in self.o_micro_clusters.values():
                    if case.id in mc.case_ids:
                        mc.case_ids.remove(case.id)
                        break
        else:
            self.all_cases.add(case.id)

        self.time += 1
        if not self.initiated:
            raise Exception

        t = self.time

        # Add point
        self.merge(case, self.time)
        self.evaluate_removing_micro_clusters(t)

    def evaluate_removing_micro_clusters(self, t: int) -> None:
        """
        Function used to calculate Tp and to evaluate
        removing any p_micro_cluster or o_micro_cluster

        Parameters
        --------------------------------------
        t: int
            A variable containing the time in single units
        """
        # Compute Tp
        try:
            part = (self.beta * self.mu) / (self.beta * self.mu - 1)
            tp = math.ceil(1 / self.lambda_ * math.log2(part))
        except:
            tp = 1
        # Test if should remove any p_micro_cluster or o_micro_cluster
        if t % tp == 0:
            for i in list(self.p_micro_clusters.keys()):
                cluster = self.p_micro_clusters[i]
                if cluster.weight < self.beta * self.mu:
                    del self.p_micro_clusters[i]

            for i in list(self.o_micro_clusters.keys()):
                cluster = self.o_micro_clusters[i]
                to = cluster.creation_time
                e = ((math.pow(2, - self.lambda_ * (t - to + tp)) - 1) /
                     (math.pow(2, - self.lambda_ * tp) - 1))
                if cluster.weight < e:
                    del self.o_micro_clusters[i]

    #    Used only in visualization
    # def is_normal(self, point: np.ndarray) -> bool:
    #    """
    #    Find if point "point" is inside any p_micro_cluster.
    #    Used only in visualization

    #    Parameters
    #    --------------------------------------
    #    point: np.ndarray
    #        Array position of a point
    #    Returns
    #    --------------------------------------
    #    True if point "point" is inside any p_micro_cluster
    #    False if point "point" is not inside any p_micro_cluster
    #    """
    #    if len(self.p_micro_clusters) == 0:
    #        return False
    #
    #    distances = [(i, self.euclidean_distance(point, cluster.centroid))
    #                 for i, cluster in self.p_micro_clusters.items()]
    #    for i, dist in distances:
    #        if dist <= self.epsilon:
    #            return True
    #    return False

    def dbscan(self, buffer: List) -> None:
        """
        Perform DBSCAN to create initial p_micro_clusters
        Works by grouping points with distance <= self._epsilon
        and filtering groups that are not dense enough (weight >= beta * mu)

        Parameters
        --------------------------------------
        buffer: List
            A buffer containing all cases
            which will be used in DBSCAN
        """
        used_cases = set()
        for case in (case for case in buffer if case.id not in used_cases):
            used_cases.add(case.id)
            group = [case]
            for other_case in (case for case in buffer if case.id not in used_cases):
                dist = self.euclidean_distance(np.array([case.graph_distance, case.time_distance]),
                                               np.array([other_case.graph_distance, other_case.time_distance]))
                if dist <= self.epsilon:
                    group.append(other_case)

            weight = len(group)
            if weight >= self.beta * self.mu:
                new_p_mc = MicroCluster(n_features=self.n_features,
                                        creation_time=0,
                                        lambda_=self.lambda_,
                                        stream_speed=self.stream_speed)
                for case in group:
                    used_cases.add(case.id)
                    new_p_mc.update(case)
                    self.all_cases.add(case.id)
                self.label += 1
                self.p_micro_clusters[self.label] = new_p_mc

            else:
                used_cases.remove(case.id)
        self.initiated = True

    def generate_clusters(self) -> Tuple[List[List], List[List]]:
        """
        Perform DBSCAN to create the final c_micro_clusters
        Works by grouping dense enough p_micro_clusters (weight >= mu)
        with distance <= 2 * self._epsilon

        Returns
        --------------------------------------
        A tuple containing two lists of groups of p-micro-clusters
        that will be inside the c-micro-cluster and which are divided
        between a dense group and a not dense group
        dense_groups: list
            A Cluster object list of dense enough
            groups of p-micro-clusters
        not_dense_groups: list
            A Cluster object list of not dense enough
            groups of p-micro-clusters
        """
        if len(self.p_micro_clusters) > 1:
            connected_clusters = []
            remaining_clusters = deque((Cluster(id=i,
                                                centroid=mc.centroid,
                                                radius=mc.radius,
                                                weight=mc.weight,
                                                case_ids=mc.case_ids)
                                        for i, mc in self.p_micro_clusters.items()))

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
                            if dist <= 2 * self.epsilon:
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
                if sum([c.weight for c in group]) >= self.mu:
                    dense_groups.append(group)
                else:
                    not_dense_groups.append(group)
            if len(dense_groups) == 0:
                dense_groups = [[]]
            if len(not_dense_groups) == 0:
                not_dense_groups = [[]]

            return dense_groups, not_dense_groups

        # only one p_micro_cluster (check if it is dense enough)
        elif len(self.p_micro_clusters) == 1:
            mc = list(self.p_micro_clusters.values())[0]
            if mc.weight >= self.mu:
                return [[Cluster(id=list(self.p_micro_clusters.keys())[0],
                                 centroid=mc.centroid,
                                 radius=mc.radius,
                                 weight=mc.weight,
                                 case_ids=mc.case_ids)]], [[]]
            else:
                return [[]], [[Cluster(id=list(self.p_micro_clusters.keys())[0],
                                       centroid=mc.centroid,
                                       radius=mc.radius,
                                       weight=mc.weight,
                                       case_ids=mc.case_ids)]]
        return [[]], [[]]

    def generate_outlier_clusters(self) -> List[Cluster]:
        """
        Generates a list of outlier clusters taking
        the o-micro-clusters stored in the dictionary

        Returns
        --------------------------------------
        List of outlier clusters
        """
        return [Cluster(id=i,
                        centroid=mc.centroid,
                        radius=mc.radius,
                        weight=mc.weight,
                        case_ids=mc.case_ids)
                for i, mc in self.o_micro_clusters.items()]
