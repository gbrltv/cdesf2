from cdesf.data_structures.cluster import Cluster
import numpy as np
import pytest


def test_initial_value():
    cluster = Cluster(0, np.array([0, 1]), 1.0, 1.0, [1, 2, 3])
    assert isinstance(cluster.id, int)
    assert cluster.id == 0
    assert isinstance(cluster.centroid, np.ndarray)
    assert np.all(cluster.centroid == [0, 1])
    assert isinstance(cluster.radius, float)
    assert cluster.radius == 1.0
    assert isinstance(cluster.weight, float)
    assert cluster.weight == 1.0
    assert isinstance(cluster.case_ids, list)
    assert cluster.case_ids == [1, 2, 3]
    assert isinstance(cluster, Cluster)


def test_no_value():
    with pytest.raises(Exception):
        assert Cluster()