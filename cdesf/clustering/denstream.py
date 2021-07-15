import math
import numpy as np
from collections import deque
from scipy.spatial import distance
from typing import Tuple, List
from ..data_structures import MicroCluster
from ..data_structures import Cluster
from ..data_structures import Case


class NoMicroClusterException(RuntimeError):
    pass


class PointNotAddedException(RuntimeError):
    pass


class DenStream:
    """
    Manages the DenStream algorithm.
    """

    def __init__(self, lambda_: float, beta: float, epsilon: float, mu: int, stream_speed: int, n_features: int):
        """
        Initializes the DenStream class and sets up the main attributes.

        Parameters
        --------------------------------------
        lambda_: float
            Sets the importance of historical data for
            the current clusters
        beta: float
            Controls micro-cluster weights and promotion
        epsilon: float
            Defines the maximum range of a micro-cluster action
        mu: int
            Controls micro-cluster weights and promotion
        stream_speed: int
            Controls how frequent the decay factor (lambda)
            influences the micro-clusters
        n_features: int
            The number of features DenStream must consider,
            in our case is always set to 2, since we have
            two attributes (graph_distance and time_distance)
        """
        self.n_features = n_features
        self.lambda_ = lambda_
        self.beta = beta
        self.epsilon = epsilon
        self.mu = mu
        self.stream_speed = stream_speed
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        self.all_cases = {}
        self.mc_id = 0
        self.time = 0
        self.no_processed_points = 0
        # Compute Tp
        try:
            self.tp = math.ceil(1 / self.lambda_ * math.log2((self.beta * self.mu) / (self.beta * self.mu - 1)))
        except Exception:
            self.tp = 1

    @staticmethod
    def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute the Euclidean Distance between two points.

        Parameters
        --------------------------------------
        point1: np.ndarray,
            Array position of a point
        point2: float,
            Cluster centroid position
        Returns
        --------------------------------------
        The Euclidean Distance: float
        """
        return distance.euclidean(point1, point2)

    def find_closest_mc(self, point: np.ndarray, micro_cluster_list: List[MicroCluster]) \
            -> Tuple[int, MicroCluster, float]:
        """
        Find the closest micro-cluster to a point according
        to the Euclidean Distance between the point and
        the cluster's centroid.

        Parameters
        --------------------------------------
        point: np.ndarray,
            Array position of a point
        micro_cluster_list: List[MicroCluster],
            The dictionary of micro-clusters
            Can be either p or o micro_clusters
        Returns
        --------------------------------------
        i: int,
            Index of the micro-cluster
        micro_cluster_list[i]: MicroCluster,
            The correspondent micro-cluster closest to the point
        dist: np.float64,
            Distance between the micro-cluster and the point
        """
        if len(micro_cluster_list) == 0:
            raise NoMicroClusterException

        distances = [(i, self.euclidean_distance(point, np.array(cluster.centroid)))
                     for i, cluster in enumerate(micro_cluster_list)]
        i, dist = min(distances, key=lambda i_dist: i_dist[1])
        return i, micro_cluster_list[i], dist

    def add_point(self, case: Case) -> int:
        """
        Tries to add a point to an existing p-micro-cluster at time "t"
        Otherwise, tries to add that point to an existing o-micro-clusters
        If all fails, creates a new o-micro-cluster with that new point

        Parameters
        --------------------------------------
        case: Case
            A Case object, with all its attributes
        Returns
        --------------------------------------
        The micro-cluster id to which the point was added
        """
        try:
            # Try to merge point with closest p_mc
            i, closest_p_mc, _ = self.find_closest_mc(case.point, self.p_micro_clusters)
            if closest_p_mc.radius_with_new_point(case.point) <= self.epsilon:
                closest_p_mc.update(case)
                return closest_p_mc.id
            else:
                raise PointNotAddedException

        except (NoMicroClusterException, PointNotAddedException):
            try:
                # Try to merge point with closest o_mc
                i, closest_o_mc, _ = self.find_closest_mc(case.point, self.o_micro_clusters)
                if closest_o_mc.radius_with_new_point(case.point) <= self.epsilon:
                    closest_o_mc.update(case)
                    # Try to promote o_micro_clusters to p_mc
                    if closest_o_mc.weight > self.beta * self.mu:
                        self.o_micro_clusters.pop(i)
                        self.p_micro_clusters.append(closest_o_mc)
                        return closest_o_mc.id
                else:
                    raise PointNotAddedException

            except (NoMicroClusterException, PointNotAddedException):
                # create new o_mc containing the new point
                new_o_mc = MicroCluster(id_=self.mc_id,
                                        n_features=self.n_features,
                                        creation_time=self.time,
                                        lambda_=self.lambda_)
                self.mc_id += 1
                new_o_mc.update(case)
                self.o_micro_clusters.append(new_o_mc)
                return new_o_mc.id
        return -1

    def decay_micro_clusters(self, micro_cluster_updated: int) -> None:
        """
        Decays micro-clusters weights after stream_speed points

        Parameters
        --------------------------------------
        micro_cluster_updated: int
            Id of the micro-cluster not to be decayed
        """
        for mc in self.p_micro_clusters:
            if mc.id != micro_cluster_updated:
                mc.decay()

    def train(self, case: Case):
        """
        "Trains" Denstream by updating micro_clusters with a new point

        Parameters
        --------------------------------------
        case: Case
            A Case object with all its attributes
        """
        # adds new point to a micro cluster and retrieves the micro cluster id
        micro_cluster_updated = self.add_point(case)
        # decays all p-micro cluster's weight except the recently updated cluster
        # self.decay_micro_clusters(micro_cluster_updated)
        # updates dictionary relating cases to micro clusters (in which mc is the case)
        self.all_cases[case.id] = micro_cluster_updated

        self.no_processed_points += 1
        # update time every stream_speed instances
        if self.no_processed_points % self.stream_speed == 0:
            self.time += 1
            # decays p-micro clusters
            for mc in self.p_micro_clusters:
                mc.decay()

        # controls the deletion of micro clusters that do not comply with the requirements
        if self.time % self.tp == 0:
            for i, mc in enumerate(self.p_micro_clusters):
                if mc.weight < self.beta * self.mu:
                    self.p_micro_clusters.pop(i)

            for i, mc in enumerate(self.o_micro_clusters):
                to = mc.creation_time
                e = ((math.pow(2, - self.lambda_ * (self.time - to + self.tp)) - 1) /
                     (math.pow(2, - self.lambda_ * self.tp) - 1))
                if mc.weight < e:
                    self.o_micro_clusters.pop(i)

    def is_normal(self, point: np.ndarray) -> bool:
        """
        Finds if a point is inside any p_micro_cluster

        Parameters
        --------------------------------------
        point: np.ndarray
            Array position of a point
        Returns
        --------------------------------------
        True if point "point" is inside any p_micro_cluster
        False if point "point" is not inside any p_micro_cluster
        """
        if len(self.p_micro_clusters) == 0:
            return False

        distances = [self.euclidean_distance(point, np.array(cluster.centroid)) for cluster in self.p_micro_clusters]
        for dist in distances:
            if dist <= self.epsilon:
                return True
        return False

    def dbscan(self, buffer: List) -> None:
        """
        Performs DBSCAN to create initial p-micro-clusters
        Works by grouping points with distance <= self._epsilon
        and filtering groups that are dense enough (weight >= beta * mu)

        Parameters
        --------------------------------------
        buffer: List
            A buffer containing all cases which will be used in DBSCAN
        """
        used_cases = set()
        for case in (case for case in buffer if case.id not in used_cases):
            used_cases.add(case.id)
            group = [case]
            for other_case in (case for case in buffer if case.id not in used_cases):
                dist = self.euclidean_distance(case.point, other_case.point)
                if dist <= self.epsilon:
                    group.append(other_case)

            weight = len(group)
            if weight >= self.beta * self.mu:
                new_p_mc = MicroCluster(id_=self.mc_id,
                                        n_features=self.n_features,
                                        creation_time=self.time,
                                        lambda_=self.lambda_)
                for case_ in group:
                    used_cases.add(case_.id)
                    new_p_mc.update(case_)
                    self.all_cases[case_.id] = new_p_mc.id
                self.mc_id += 1
                self.p_micro_clusters.append(new_p_mc)
            else:
                used_cases.remove(case.id)

    def generate_clusters(self) -> Tuple[List[List], List[List]]:
        """
        Perform DBSCAN to create the final micro-clusters
        Works by grouping dense enough p_micro_clusters (weight >= mu)
        with distance <= 2 * self._epsilon

        Returns
        --------------------------------------
        A tuple containing two lists of groups of p-micro-clusters
        that will be inside the c-micro-cluster and which are divided
        between a dense group and a not dense group
        dense_groups: list
            A Cluster object list of dense enough groups of c-micro-clusters
        not_dense_groups: list
            A Cluster object list of not dense enough groups of p-micro-clusters
        """
        if len(self.p_micro_clusters) > 1:
            connected_clusters = []
            remaining_clusters = deque((Cluster(id_=i,
                                                centroid=mc.centroid,
                                                radius=mc.radius,
                                                weight=mc.weight,
                                                case_ids=[k for k, v in self.all_cases.items() if v == mc.id])
                                        for i, mc in enumerate(self.p_micro_clusters)))

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
                    # try to add remaining clusters to the existing group as it is
                    # if we add a new cluster to that group perform the check again
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
            mc = self.p_micro_clusters[0]
            if mc.weight >= self.mu:
                return [[Cluster(id_=mc.id,
                                 centroid=mc.centroid,
                                 radius=mc.radius,
                                 weight=mc.weight,
                                 case_ids=[k for k, v in self.all_cases.items() if v == mc.id])]], [[]]
            else:
                return [[]], [[Cluster(id_=mc.id,
                                       centroid=mc.centroid,
                                       radius=mc.radius,
                                       weight=mc.weight,
                                       case_ids=[k for k, v in self.all_cases.items() if v == mc.id])]]
        return [[]], [[]]

    def generate_outlier_clusters(self) -> List[Cluster]:
        """
        Generates a list of outlier clusters taking
        the o-micro-clusters stored in the dictionary

        Returns
        --------------------------------------
        List of outlier clusters
        """
        return [Cluster(id_=mc.id,
                        centroid=mc.centroid,
                        radius=mc.radius,
                        weight=mc.weight,
                        case_ids=[k for k, v in self.all_cases.items() if v == mc.id])
                for mc in self.o_micro_clusters]
