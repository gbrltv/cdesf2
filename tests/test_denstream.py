from cdesf.clustering import DenStream, NoMicroClusterException
from cdesf.data_structures import MicroCluster
from cdesf.data_structures import Cluster
from cdesf.data_structures import Case
from cdesf.utils import calculate_case_distances, initialize_graph
from math import log10
import networkx as nx
from datetime import datetime
import numpy as np
import pytest


class TestDenstream:
    @pytest.fixture
    def denstream(self):
        denstream = DenStream(
            lambda_=0.15, beta=0.3, epsilon=0.1, mu=4, stream_speed=1000, n_features=2
        )
        return denstream

    @pytest.fixture
    def cases_list(self):
        case_list = []
        case = Case("1")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_list.append(case)

        case = Case("2")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case_list.append(case)

        case = Case("3")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            }
        )
        case_list.append(case)

        case = Case("4")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_list.append(case)

        return case_list

    def test_initial_value(self, denstream):
        assert isinstance(denstream, DenStream)
        assert denstream.n_features == 2
        assert denstream.lambda_ == 0.15
        assert denstream.beta == 0.3
        assert denstream.epsilon == 0.1
        assert denstream.mu == 4
        assert denstream.stream_speed == 1000
        assert denstream.p_micro_clusters == []
        assert denstream.o_micro_clusters == []
        assert denstream.time == 0
        assert denstream.all_cases == {}

    # This is a compile-time error.
    # def test_no_value(self):
    #     with pytest.raises(Exception):
    #         assert DenStream()

    def test_euclidean_distance(self):
        point_1 = np.array([0, 0])
        point_2 = np.array([0, 1])
        euc_dist = DenStream.euclidean_distance(point_1, point_2)
        assert isinstance(euc_dist, np.float64)
        assert euc_dist == 1

    def test_find_closest_mc(self, denstream):
        point = np.array([0, 0])

        with pytest.raises(NoMicroClusterException):
            assert denstream.find_closest_mc(point, denstream.p_micro_clusters)
        with pytest.raises(NoMicroClusterException):
            assert denstream.find_closest_mc(point, denstream.o_micro_clusters)

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([0, 1])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([0, 1])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)

        i, mc, dist = denstream.find_closest_mc(point, denstream.p_micro_clusters)
        assert i == 0
        assert np.all(mc.cf1 == [0, 1])
        assert mc.weight == 2
        assert dist == 0.5

        point = np.array([0, 2])
        micro_cluster = MicroCluster(2, 2, 1, 0.15)
        micro_cluster.cf1 = np.array([0, 1])
        micro_cluster.weight = 2
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(3, 2, 1, 0.15)
        micro_cluster.cf1 = np.array([0, 1])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)

        i, mc, dist = denstream.find_closest_mc(point, denstream.p_micro_clusters)
        assert i == 1
        assert np.all(mc.cf1 == [0, 1])
        assert mc.weight == 1
        assert dist == 1

        micro_cluster = MicroCluster(4, 2, 1, 0.15)
        micro_cluster.cf1 = np.array([0, 3])
        micro_cluster.weight = 2
        denstream.o_micro_clusters = []
        denstream.o_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(5, 2, 1, 0.15)
        micro_cluster.cf1 = np.array([0, 1])
        micro_cluster.weight = 3
        denstream.o_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(6, 2, 1, 0.15)
        micro_cluster.cf1 = np.array([0, 2])
        micro_cluster.weight = 1
        denstream.o_micro_clusters.append(micro_cluster)

        point = np.array([0, 3])
        i, mc, dist = denstream.find_closest_mc(point, denstream.o_micro_clusters)
        assert i == 2
        assert np.all(mc.cf1 == [0, 2])
        assert mc.weight == 1

    def test_add_point(self, denstream: DenStream):
        case_list = []
        case_1 = Case("1")
        case_1.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_1.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case_1.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_list.append(case_1)

        case_2 = Case("2")
        case_2.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_2.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case_list.append(case_2)

        graph = initialize_graph(case_list)

        case_3 = Case("3")
        case_3.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_3.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case_3.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_3.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            }
        )

        case_3.distances = calculate_case_distances(graph, case_3)

        micro_cluster = MicroCluster(10, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([0.5, -0.5])
        micro_cluster.cf2 = np.array([0.5, -0.1])
        micro_cluster.weight = 10
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(11, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([0.0, 0.0])
        micro_cluster.cf2 = np.array([0.0, 0.0])
        micro_cluster.weight = 5
        denstream.o_micro_clusters.append(micro_cluster)
        denstream.mc_id = 2

        mc_id = denstream.add_point(case_3)
        assert mc_id == 2
        assert len(denstream.o_micro_clusters) == 2
        assert denstream.o_micro_clusters[1].radius == 0
        assert denstream.o_micro_clusters[1].weight == 1
        assert np.all(denstream.o_micro_clusters[1].cf1 == case_3.point)
        assert np.all(denstream.o_micro_clusters[1].cf2 == case_3.point * case_3.point)

        cf = denstream.o_micro_clusters[1].cf1.copy()
        cf2 = denstream.o_micro_clusters[1].cf2.copy()
        mc_id = denstream.add_point(case_3)
        assert mc_id == 2
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 2
        assert denstream.p_micro_clusters[1].weight == 2
        assert np.all(denstream.p_micro_clusters[1].cf1 == cf + case_3.point)
        assert np.all(
            denstream.p_micro_clusters[1].cf2 == cf2 + case_3.point * case_3.point
        )

    def test_decay_micro_clusters(self, denstream):
        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([5.0, 5.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 5
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 5.0])
        micro_cluster.cf2 = np.array([3.0, 0.0])
        micro_cluster.weight = 10
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(2, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([0.0, 0.0])
        micro_cluster.cf2 = np.array([10.0, 2.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters.append(micro_cluster)

        denstream.decay_micro_clusters(0)
        assert np.all(denstream.p_micro_clusters[0].cf1 == np.array([5, 5]))
        assert np.all(denstream.p_micro_clusters[0].cf2 == np.array([1, 1]))
        assert np.all(denstream.p_micro_clusters[0].weight == 5)

        assert np.all(
            denstream.p_micro_clusters[1].cf1 == np.array([1, 5]) * 2 ** (-0.15)
        )
        assert np.all(
            denstream.p_micro_clusters[1].cf2 == np.array([3, 0]) * 2 ** (-0.15)
        )
        assert np.all(denstream.p_micro_clusters[1].weight == 10 * 2 ** (-0.15))

        assert np.all(
            denstream.p_micro_clusters[2].cf1 == np.array([0, 0]) * 2 ** (-0.15)
        )
        assert np.all(
            denstream.p_micro_clusters[2].cf2 == np.array([10, 2]) * 2 ** (-0.15)
        )
        assert np.all(denstream.p_micro_clusters[2].weight == 3 * 2 ** (-0.15))

    def test_dbscan(self, denstream: DenStream, cases_list):
        graph = initialize_graph(cases_list)
        for case in cases_list:
            case.distances = calculate_case_distances(graph, case)

        denstream.dbscan(cases_list)
        assert len(denstream.p_micro_clusters) == 1
        assert denstream.all_cases.keys() == {"1", "4"}
        new_p_mc = denstream.p_micro_clusters[0]
        assert new_p_mc.weight == 2
        assert new_p_mc.creation_time == 0
        assert new_p_mc.lambda_ == 0.15

    def test_train(self, denstream: DenStream):
        # anomalous_denstream = DenStream(2, 0.15, 0.3, 0.1, 4, 1000)

        cases_list = []

        case = Case("5")
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            }
        )
        case.add_event(
            {
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 40),
            }
        )
        cases_list.append(case)

        case = Case("3")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            }
        )
        cases_list.append(case)

        case = Case("1")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        cases_list.append(case)

        case = Case("4")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        cases_list.append(case)

        case = Case("2")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        cases_list.append(case)

        pmg = initialize_graph(cases_list)
        for case in cases_list:
            case.distances = calculate_case_distances(pmg, case)

        denstream.dbscan(cases_list)

        assert denstream.stream_speed == 1000
        assert denstream.no_processed_points == 0
        assert denstream.time == 0
        assert len(denstream.p_micro_clusters) == 1
        assert len(denstream.o_micro_clusters) == 0

        case_5 = cases_list[0]
        denstream.train(case_5)
        assert denstream.all_cases == {"1": 0, "4": 0, "5": 1}
        assert denstream.time == 0
        assert denstream.no_processed_points == 1
        assert not denstream.p_micro_clusters == []
        assert not denstream.o_micro_clusters == []
        assert len(denstream.p_micro_clusters) == 1
        assert len(denstream.o_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [0.25, 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [0.03125, 0])
        assert denstream.p_micro_clusters[0].weight == 2
        assert denstream.p_micro_clusters[0].creation_time == 0

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        case_3 = cases_list[1]
        denstream.train(case_3)
        assert denstream.all_cases == {"1": 0, "3": 0, "4": 0, "5": 1}
        assert denstream.time == 0
        assert denstream.no_processed_points == 2
        assert len(denstream.p_micro_clusters) == 1
        assert len(denstream.o_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [0.5, 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [0.09375, 0])
        assert denstream.p_micro_clusters[0].weight == 3
        assert denstream.p_micro_clusters[0].creation_time == 0

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        case_1 = cases_list[2]
        denstream.train(case_1)
        assert denstream.all_cases == {"1": 0, "3": 0, "4": 0, "5": 1}
        assert denstream.time == 0
        assert denstream.no_processed_points == 3
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [(7 / 64), 0])
        assert denstream.p_micro_clusters[0].weight == 4

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        case_4 = cases_list[3]
        denstream.train(case_4)
        assert denstream.all_cases == {"1": 0, "3": 0, "4": 0, "5": 1}
        assert denstream.time == 0
        assert denstream.no_processed_points == 4
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [0.75, 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [0.125, 0])
        assert denstream.p_micro_clusters[0].weight == 5

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        case_1 = cases_list[4]
        denstream.train(case_1)
        assert denstream.all_cases == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1}
        assert denstream.time == 0
        assert denstream.no_processed_points == 5
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [0.75, 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [0.125, 0])
        assert denstream.p_micro_clusters[0].weight == 6

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        case = Case("6")
        case.add_event(
            {
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 40),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 50),
            }
        )

        case.distances = calculate_case_distances(pmg, case)
        cases_list.insert(0, case)

        denstream.train(cases_list[0])
        assert denstream.all_cases == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2}
        assert denstream.time == 0
        assert denstream.no_processed_points == 6
        assert len(denstream.o_micro_clusters) == 2
        assert len(denstream.p_micro_clusters) == 1

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == [0.75, 0])
        assert np.all(denstream.p_micro_clusters[0].cf2 == [0.125, 0])
        assert denstream.p_micro_clusters[0].weight == 6

        assert denstream.o_micro_clusters[0].id == 1
        assert np.all(denstream.o_micro_clusters[0].cf1 == [(5 / 8), 0])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [(5 / 8) ** 2, 0])
        assert denstream.o_micro_clusters[0].weight == 1
        assert denstream.o_micro_clusters[0].creation_time == 0

        assert denstream.o_micro_clusters[1].id == 2
        assert np.all(denstream.o_micro_clusters[1].cf1 == [1, 1])
        assert np.all(denstream.o_micro_clusters[1].cf2 == [1, 1])
        assert denstream.o_micro_clusters[1].weight == 1
        assert denstream.o_micro_clusters[1].creation_time == 0
        assert denstream.all_cases == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2}

        denstream.stream_speed = 7
        denstream.train(cases_list[0])
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 2

        CF = np.array([0.75, 0]) * (2 ** (-0.15))
        CF2 = np.array([0.125, 0]) * (2 ** (-0.15))
        weight = 6 * (2 ** (-0.15))

        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].cf1 == CF)
        assert np.all(denstream.p_micro_clusters[0].cf2 == CF2)
        assert denstream.p_micro_clusters[0].weight == weight

        CF = (np.array([1, 1]) + cases_list[0].point) * (2 ** (-0.15))
        CF2 = (np.array([1, 1]) + (cases_list[0].point * cases_list[0].point)) * (
            2 ** (-0.15)
        )
        weight = (1 + 1) * (2 ** (-0.15))

        assert denstream.p_micro_clusters[1].id == 2
        assert np.all(denstream.p_micro_clusters[1].cf1 == CF)
        assert np.all(denstream.p_micro_clusters[1].cf2 == CF2)
        assert denstream.p_micro_clusters[1].weight == weight

        for i in range(16):
            assert denstream.all_cases == {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 1,
                "6": 2,
            }
            denstream.stream_speed = 8 + i
            # mc.weight > self.beta * self.mu
            denstream.train(cases_list[0])
            assert denstream.time == i + 2
            assert denstream.p_micro_clusters[0].id == 0
            CF = np.array([0.75, 0]) * (2 ** (-0.15))
            CF2 = np.array([0.125, 0]) * (2 ** (-0.15))
            weight = 6 * (2 ** (-0.15))
            for _ in range(i + 1):
                CF *= 2 ** (-0.15)
                CF2 *= 2 ** (-0.15)
                weight *= 2 ** (-0.15)
            assert np.all(denstream.p_micro_clusters[0].cf1 == CF)
            assert np.all(denstream.p_micro_clusters[0].cf2 == CF2)
            assert denstream.p_micro_clusters[0].weight == weight

            assert denstream.p_micro_clusters[1].id == 2
            CF_1 = (np.array([1, 1]) + cases_list[0].point) * (2 ** (-0.15))
            CF2_1 = (np.array([1, 1]) + (cases_list[0].point * cases_list[0].point)) * (
                2 ** (-0.15)
            )
            weight_1 = (1 + 1) * (2 ** (-0.15))
            for _ in range(i + 1):
                CF_1 += cases_list[0].point
                CF_1 *= 2 ** (-0.15)
                CF2_1 += cases_list[0].point * cases_list[0].point
                CF2_1 *= 2 ** (-0.15)
                weight_1 += 1
                weight_1 *= 2 ** (-0.15)
            assert np.all(denstream.p_micro_clusters[1].cf1 == CF_1)
            assert np.all(denstream.p_micro_clusters[1].cf2 == CF2_1)
            assert denstream.p_micro_clusters[1].weight == weight_1

        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 2

        assert denstream.time == 17
        assert denstream.no_processed_points == 23
        assert denstream.stream_speed == 23
        denstream.stream_speed += 1

        case = Case("7")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityG",
                "time:timestamp": datetime(2015, 5, 10, 8, 1, 00),
            }
        )
        case.distances = calculate_case_distances(pmg, case)
        cases_list.insert(0, case)
        assert cases_list[0].id == "7"

        denstream.train(cases_list[0])
        assert denstream.all_cases == {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 1,
            "6": 2,
            "7": 3,
        }
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 1
        assert denstream.p_micro_clusters[0].id == 2
        CF_1 *= 2 ** (-0.15)
        CF2_1 *= 2 ** (-0.15)
        weight_1 *= 2 ** (-0.15)
        assert np.all(denstream.p_micro_clusters[0].cf1 == CF_1)
        assert np.all(denstream.p_micro_clusters[0].cf2 == CF2_1)
        assert denstream.p_micro_clusters[0].weight == weight_1

        assert denstream.o_micro_clusters[0].id == 3
        assert np.all(denstream.o_micro_clusters[0].cf1 == [1, log10(60)])
        assert np.all(denstream.o_micro_clusters[0].cf2 == [1, log10(60) ** 2])
        assert denstream.o_micro_clusters[0].weight == 1

    def test_generate_clusters(self, denstream):
        assert len(denstream.p_micro_clusters) == 0

        dense_group, not_dense_group = denstream.generate_clusters()
        assert dense_group == [[]]
        assert not_dense_group == [[]]

        case_0 = Case("0")
        case_1 = Case("1")

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([5.0, 5.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 5
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        assert len(denstream.p_micro_clusters) == 1
        assert micro_cluster.weight >= denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert not dense_group == [[]]
        assert not_dense_group == [[]]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 5
        assert cluster.case_ids == ["0", "1"]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([3.0, 3.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        assert len(denstream.p_micro_clusters) == 1
        assert micro_cluster.weight < denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert dense_group == [[]]
        assert not not_dense_group == [[]]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 3
        assert cluster.case_ids == ["0", "1"]

        case_2 = Case("2")
        case_3 = Case("3")

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([3.0, 3.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([4.0, 4.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            <= 2 * denstream.epsilon
        )
        assert cl1.weight + cl2.weight >= denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert not dense_group == [[]]
        assert not_dense_group == [[]]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 3
        assert cluster.case_ids == ["0", "1"]
        cluster = cluster_list[1]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 4
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert dense_group[1]
            assert cluster_list[2]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([3.0, 3.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 1
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([3.0, 3.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            <= 2 * denstream.epsilon
        )
        assert cl1.weight + cl2.weight < denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert dense_group == [[]]
        assert not not_dense_group == [[]]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [3, 3])
        assert cluster.radius == 0
        assert cluster.weight == 1
        assert cluster.case_ids == ["0", "1"]
        cluster = cluster_list[1]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [3, 3])
        assert cluster.radius == 0
        assert cluster.weight == 1
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert cluster_list[2]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([3.0, 3.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 1.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            > 2 * denstream.epsilon
        )
        assert cl1.weight < denstream.mu
        assert cl2.weight >= denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert not dense_group == [[]]
        assert not not_dense_group == [[]]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 3
        assert cluster.case_ids == ["0", "1"]
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.25, 0.25])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 4
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert dense_group[1]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([6.0, 6.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 6
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 1.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            > 2 * denstream.epsilon
        )
        assert cl1.weight >= denstream.mu
        assert cl2.weight < denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert not dense_group == [[]]
        assert not not_dense_group == [[]]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 6
        assert cluster.case_ids == ["0", "1"]
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.5, 0.5])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 2
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert dense_group[1]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([6.0, 6.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 6
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 1.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            > 2 * denstream.epsilon
        )
        assert cl1.weight >= denstream.mu
        assert cl2.weight >= denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert not dense_group == [[]]
        assert not_dense_group == [[]]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 6
        assert cluster.case_ids == ["0", "1"]
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = dense_group[1]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.25, 0.25])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 4
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert dense_group[2]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 1.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.cf1 = np.array([1.0, 1.0])
        micro_cluster.cf2 = np.array([1.0, 1.0])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert (
            denstream.euclidean_distance(cl1.centroid, cl2.centroid)
            > 2 * denstream.epsilon
        )
        assert cl1.weight < denstream.mu
        assert cl2.weight < denstream.mu
        dense_group, not_dense_group = denstream.generate_clusters()

        assert dense_group == [[]]
        assert not not_dense_group == [[]]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 0
        assert np.all(cluster.centroid == [(1 / 3), (1 / 3)])
        assert cluster.radius == cl1.radius
        assert cluster.weight == 3
        assert cluster.case_ids == ["0", "1"]
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = not_dense_group[1]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.5, 0.5])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 2
        assert cluster.case_ids == ["2", "3"]
        with pytest.raises(IndexError):
            assert not_dense_group[2]
            assert cluster_list[1]

    def test_generate_outlier_clusters(self, denstream):
        assert denstream.generate_outlier_clusters() == []

        case_0 = Case("0")
        case_1 = Case("1")
        case_2 = Case("2")
        case_3 = Case("3")

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.weight = 3
        denstream.o_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.weight = 4
        denstream.o_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(3, 2, 0, 0.15)
        micro_cluster.weight = 6
        denstream.o_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        assert denstream.all_cases == {"0": 0, "1": 1, "2": 3, "3": 3}

        cluster_list = denstream.generate_outlier_clusters()

        cluster = cluster_list[0]
        assert cluster.id == 0
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 3
        assert cluster.case_ids == ["0"]

        cluster = cluster_list[1]
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 4
        assert cluster.case_ids == ["1"]

        cluster = cluster_list[2]
        assert cluster.id == 3
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 6
        assert cluster.case_ids == ["2", "3"]
