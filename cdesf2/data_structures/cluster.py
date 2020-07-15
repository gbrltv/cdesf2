from numpy import ndarray


class Cluster:
    """
    Class that represents a cluster.
    """

    def __init__(self, id: int, centroid: ndarray, radius: float, weight: float, case_ids: set):
        """
        Receives an identifier, the position of the centroid,
        the radius and a set of case identifier and initializes a cluster.

        Parameters
        --------------------------------------
        id: int
            Cluster identifier
        centroid: ndarray
            Cluster centroid position
        radius: float
            Cluster radius,
            measure of how far is the cluster influence
        weight: float
            CLuster weight
        case_ids: set
            Set of case identifiers inside that cluster
        """
        self.id = id
        self.centroid = centroid
        self.radius = radius
        self.weight = weight
        self.case_ids = case_ids
