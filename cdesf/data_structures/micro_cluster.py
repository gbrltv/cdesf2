import numpy as np
from ..data_structures import Case


class MicroCluster:
    """
    The class represents a micro-cluster and its attributes.
    """

    def __init__(self, id_: int, n_features: int, creation_time: int, lambda_: float):
        """
        Receives the number of features analyzed, the creation time,
        the importance of historical data and the stream speed.
        Initializes the MicroCluster attributes.

        Parameters
        --------------------------------------
        n_features: int
            The number of features DenStream must consider,
            in our case is always set to 2, since we have
            two attributes (graph_distance and time_distance)
        creation_time: int
            Creation time in single units
        lambda_: float
            Sets the importance of historical data for the
            current clusters
        """
        self.id = id_
        self.cf1 = np.zeros(n_features)
        self.cf2 = np.zeros(n_features)
        self.weight = 0
        self.lambda_ = lambda_
        self.factor = 2 ** (-lambda_)
        self.creation_time = creation_time

    @property
    def centroid(self) -> np.ndarray:
        """
        Computes the micro-cluster's centroid value,
        which is given by CF divided by weight.

        Returns
        --------------------------------------
        The micro-cluster's centroid value
        """
        return self.cf1 / self.weight

    @property
    def radius(self) -> np.ndarray:
        """
        Computes the micro-cluster's radius.

        Returns
        --------------------------------------
        The micro-cluster's radius
        """
        weighted_cf1 = self.cf1 / self.weight
        weighted_cf2 = self.cf2 / self.weight

        cf_operation = np.maximum(weighted_cf2 - (weighted_cf1 ** 2), 0)
        cf_root = np.sqrt(cf_operation)

        return np.nanmax(cf_root)

    def radius_with_new_point(self, point: np.ndarray) -> np.ndarray:
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
        weighted_cf1 = (self.cf1 + point) / (self.weight + 1)
        weighted_cf2 = (self.cf2 + (point ** 2)) / (self.weight + 1)

        cf_operation = np.maximum(weighted_cf2 - (weighted_cf1 ** 2), 0)
        cf_root = np.sqrt(cf_operation)

        return np.nanmax(cf_root)

    def update(self, case: Case):
        """
        Updates the micro-cluster given a new case

        Parameters
        --------------------------------------
        case: Case
            A case object
        """
        self.weight += 1

        self.cf1 += case.point
        self.cf2 += case.point ** 2

    def decay(self):
        """
        Decays a micro-cluster using the factor, which is based on lambda
        """
        self.weight *= self.factor

        self.cf1 *= self.factor
        self.cf2 *= self.factor
