import numpy as np
from ..data_structures import Case


class MicroCluster:
    """
    The class represents a micro-cluster and its attributes.
    """

    def __init__(self, id_: int, n_features: int, creation_time: int, lambda_: float):
        """
        Receives the number of feature analyzed, the creation time,
        the importance of historical data and the stream speed.
        Initializes the MicroCluster attributes.

        Parameters
        --------------------------------------
        n_features: int
            The number of features DenStream must consider,
            in our case is always set to 2, since we have
            two attributes (graph_distance and time_distance)
        time: int
            Creation time in single units
        lambda_: float
            Sets the importance of historical data for the
            current clusters
        """
        self.id = id_
        self.CF = np.zeros(n_features)
        self.CF2 = np.zeros(n_features)
        self.weight = 0
        self.lambda_ = lambda_
        self.factor = 2 ** (-self.lambda_)
        self.creation_time = creation_time

    @property
    def centroid(self) -> float:
        """
        Computes the micro-cluster's centroid value,
        which is given by CF divided by weight.

        Returns
        --------------------------------------
        The micro-cluster's centroid value
        """
        return self.CF / self.weight

    @property
    def radius(self) -> float:
        """
        Computes the micro-cluster's radius.

        Returns
        --------------------------------------
        The micro-cluster's radius
        """
        cf1_squared = (self.CF / self.weight) ** 2
        return np.nan_to_num(np.nanmax(((self.CF2 / self.weight) - cf1_squared) ** (1 / 2)))

    def radius_with_new_point(self, point: np.ndarray) -> float:
        """
        Computes the micro-cluster's radius considering a new point.
        The returned value is then compared to self._epsilon to check
        whether the point must be absorbed or not.

        Parameters
        --------------------------------------
        point: np.ndarray
            Contain graph_distance and time_distance
        Returns
        --------------------------------------
        The micro-cluster's radius
        """
        cf1 = self.CF + point
        cf2 = self.CF2 + point * point
        weight = self.weight + 1

        cf1_squared = (cf1 / weight) ** 2
        return np.nan_to_num(np.nanmax(((cf2 / weight) - cf1_squared) ** (1 / 2)))

    def update(self, case: Case):
        """
        Updates the micro-cluster given a new case

        Parameters
        --------------------------------------
        case: Case
            A case object
        """
        point = case.point
        self.CF += point
        self.CF2 += point * point
        self.weight += 1

    def decay(self):
        """
        Decays a micro-cluster using the factor, which is based on lambda
        """
        self.CF *= self.factor
        self.CF2 *= self.factor
        self.weight *= self.factor
