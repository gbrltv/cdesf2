import numpy as np


class Cluster:
    """
    Class that represents a cluster.
    """

    def __init__(self, id_: int, centroid: np.ndarray, radius: float, weight: float, case_ids: list):
        """
        Receives an identifier, the position of the centroid,
        the radius and a set of case identifier and initializes a cluster.

        Parameters
        --------------------------------------
        id_: int
            Cluster identifier
        centroid: np.ndarray
            Cluster centroid position
        radius: float
            Cluster radius,
            measure of how far is the cluster influence
        weight: float
            CLuster weight
        case_ids: set
            Set of case identifiers inside that cluster
        """
        self.id = id_
        self.centroid = centroid
        self.radius = radius
        self.weight = weight
        self.case_ids = case_ids

    def __str__(self):
        return f'ID: {self.id} | Centroid: {self.centroid} | Radius: {self.radius} | Weight: {self.weight}'
