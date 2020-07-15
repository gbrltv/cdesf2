import numpy as np
from math import sqrt
from ..data_structures import Case
from typing import Optional


class MicroCluster:
    """
    The class represents a micro-cluster and its attributes.
    """

    def __init__(self, n_features: int, creation_time: int, lambda_: float, stream_speed: int):
        """
        Receives the number of feature analyzed, the creation time,
        the importance of histarical data and the streem speed.
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
        stream_speed: int
            Speed of stream
        """
        self.CF = np.zeros(n_features)
        self.CF2 = np.zeros(n_features)
        self.weight = 0
        self.creation_time = creation_time
        self.case_ids = set()
        self.lambda_ = lambda_
        self.stream_speed = stream_speed
        self.count_to_decay = self.stream_speed

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
        a = np.sqrt(np.sum(np.square(self.CF2))) / self.weight
        b = np.square(np.sqrt(np.sum(np.square(self.CF))) / self.weight)
        s = a - b
        if s < 0:
            s = 0
        return sqrt(s)

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

        a = np.sqrt(np.sum(np.square(cf2))) / weight
        b = np.square(np.sqrt(np.sum(np.square(cf1))) / weight)
        s = a - b
        if s < 0:
            s = 0
        return sqrt(s)

    def update(self, case: Optional[Case]):
        """
        Updates the micro-cluster weights either
        considering a new case or not.

        Parameters
        --------------------------------------
        case: Optional[Case]
            A case object
        """
        if case is None:
            factor = 2 ** (-self.lambda_)
            self.CF *= factor
            self.CF2 *= factor
            self.weight *= factor
        else:
            case_point = np.array([case.graph_distance,
                                  case.time_distance])
            self.CF += case_point
            self.CF2 += case_point * case_point
            self.weight += 1
            self.case_ids.add(case.id)
