from cdesf2.data_structures.cluster import Cluster
import numpy as np
import pytest


def test_initial_value():
    centroid = np.array([0, 1])
    case_ids = set()
    case_ids.add(1)
    case_ids.add(2)
    case_ids.add(3)
    cluster = Cluster(1, centroid, 1.0, 1.0, case_ids)
    assert type(cluster.id) is int
    assert cluster.id == 1
    assert type(cluster.centroid) is np.ndarray
    assert np.all(cluster.centroid == [0, 1])
    assert type(cluster.radius) is float
    assert cluster.radius == 1.0
    assert type(cluster.weight) is float
    assert cluster.weight == 1.0
    assert type(cluster.case_ids) is set
    assert cluster.case_ids == {1, 2, 3}
    assert type(cluster) is Cluster


def test_no_value():
    with pytest.raises(Exception):
        assert Cluster()