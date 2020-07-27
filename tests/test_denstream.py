from cdesf2.clustering import DenStream, NoMicroClusterException
from cdesf2.data_structures import MicroCluster
from cdesf2.data_structures import Cluster
from cdesf2.data_structures import Case
from cdesf2.utils import extract_case_distances, initialize_graph
from math import ceil
import networkx as nx
from datetime import datetime
import numpy as np
import pytest


class TestDenstream:
    @pytest.fixture
    def denstream(self):
        denstream = DenStream(2, 0.15, 0.3, 0.1, 4, 1000)
        return denstream

    @pytest.fixture
    def cases_list(self):
        case_list = []
        case = Case('1')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_list.append(case)

        case = Case('2')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_list.append(case)

        case = Case('3')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case.set_activity('activityD', datetime(2015, 5, 10, 8, 00, 30))
        case_list.append(case)

        case = Case('4')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_list.append(case)

        return case_list

    def test_initial_value_den(self, denstream):
        assert isinstance(denstream, DenStream)
        assert denstream.n_features == 2
        assert denstream.lambda_ == 0.15
        assert denstream.beta == 0.3
        assert denstream.epsilon == 0.1
        assert denstream.mu == 4
        assert denstream.p_micro_clusters == []
        assert denstream.o_micro_clusters == []
        assert denstream.time == 0
        assert denstream.all_cases == {}
        assert denstream.stream_speed == 1000

    def test_no_value_den(self):
        with pytest.raises(Exception):
            assert DenStream()

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
        micro_cluster.CF = np.array([0, 1])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([0, 1])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)

        i, mc, dist = denstream.find_closest_mc(point, denstream.p_micro_clusters)
        assert i == 0
        assert np.all(mc.CF == [0, 1])
        assert mc.weight == 2
        assert dist == 0.5

        point = np.array([0, 2])
        micro_cluster = MicroCluster(2, 2, 1, 0.15)
        micro_cluster.CF = np.array([0, 1])
        micro_cluster.weight = 2
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(3, 2, 1, 0.15)
        micro_cluster.CF = np.array([0, 1])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)

        i, mc, dist = denstream.find_closest_mc(point, denstream.p_micro_clusters)
        assert i == 1
        assert np.all(mc.CF == [0, 1])
        assert mc.weight == 1
        assert dist == 1

        micro_cluster = MicroCluster(4, 2, 1, 0.15)
        micro_cluster.CF = np.array([0, 3])
        micro_cluster.weight = 2
        denstream.o_micro_clusters = []
        denstream.o_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(5, 2, 1, 0.15)
        micro_cluster.CF = np.array([0, 1])
        micro_cluster.weight = 3
        denstream.o_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(6, 2, 1, 0.15)
        micro_cluster.CF = np.array([0, 2])
        micro_cluster.weight = 1
        denstream.o_micro_clusters.append(micro_cluster)

        point = np.array([0, 3])
        i, mc, dist = denstream.find_closest_mc(point, denstream.o_micro_clusters)
        assert i == 2
        assert np.all(mc.CF == [0, 2])
        assert mc.weight == 1

    def test_add_point(self, denstream):
        case_list = []
        case_1 = Case('1')
        case_1.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_1.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_1.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_list.append(case_1)

        case_2 = Case('2')
        case_2.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_2.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_list.append(case_2)

        graph = nx.DiGraph()
        graph = initialize_graph(graph, case_list)

        case_3 = Case('3')
        case_3.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_3.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_3.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_3.set_activity('activityD', datetime(2015, 5, 10, 8, 00, 30))

        trace_distance, time_distance = extract_case_distances(graph, case_3)
        case_3.graph_distance = trace_distance
        case_3.time_distance = time_distance

        micro_cluster = MicroCluster(10, 2, 0, 0.15)
        micro_cluster.CF = np.array([0.5, -0.5])
        micro_cluster.CF2 = np.array([0.5, -0.1])
        micro_cluster.weight = 10
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(11, 2, 0, 0.15)
        micro_cluster.CF = np.array([0.0, 0.0])
        micro_cluster.CF2 = np.array([0.0, 0.0])
        micro_cluster.weight = 5
        denstream.o_micro_clusters.append(micro_cluster)
        denstream.mc_id = 2

        mc_id = denstream.add_point(case_3)
        assert mc_id == 2
        assert len(denstream.o_micro_clusters) == 2
        assert denstream.o_micro_clusters[1].radius == 0
        assert denstream.o_micro_clusters[1].weight == 1
        assert np.all(denstream.o_micro_clusters[1].CF == case_3.point)
        assert np.all(denstream.o_micro_clusters[1].CF2 == case_3.point * case_3.point)

        cf = denstream.o_micro_clusters[1].CF.copy()
        cf2 = denstream.o_micro_clusters[1].CF2.copy()
        mc_id = denstream.add_point(case_3)
        assert mc_id == 2
        assert len(denstream.o_micro_clusters) == 1
        assert len(denstream.p_micro_clusters) == 2
        assert denstream.p_micro_clusters[1].weight == 2
        assert np.all(denstream.p_micro_clusters[1].CF == cf + case_3.point)
        assert np.all(denstream.p_micro_clusters[1].CF2 == cf2 + case_3.point * case_3.point)

    def test_decay_micro_clusters(self, denstream):
        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([5.0, 5.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 5
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 5.0])
        micro_cluster.CF2 = np.array([3.0, 0.0])
        micro_cluster.weight = 10
        denstream.p_micro_clusters.append(micro_cluster)

        micro_cluster = MicroCluster(2, 2, 0, 0.15)
        micro_cluster.CF = np.array([0.0, 0.0])
        micro_cluster.CF2 = np.array([10.0, 2.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters.append(micro_cluster)

        denstream.decay_micro_clusters(0)
        assert np.all(denstream.p_micro_clusters[0].CF == np.array([5, 5]))
        assert np.all(denstream.p_micro_clusters[0].CF2 == np.array([1, 1]))
        assert np.all(denstream.p_micro_clusters[0].weight == 5)

        assert np.all(denstream.p_micro_clusters[1].CF == np.array([1, 5]) * 2 ** (-0.15))
        assert np.all(denstream.p_micro_clusters[1].CF2 == np.array([3, 0]) * 2 ** (-0.15))
        assert np.all(denstream.p_micro_clusters[1].weight == 10 * 2 ** (-0.15))

        assert np.all(denstream.p_micro_clusters[2].CF == np.array([0, 0]) * 2 ** (-0.15))
        assert np.all(denstream.p_micro_clusters[2].CF2 == np.array([10, 2]) * 2 ** (-0.15))
        assert np.all(denstream.p_micro_clusters[2].weight == 3 * 2 ** (-0.15))

    def test_dbscan(self, denstream, cases_list):
        graph = nx.DiGraph()
        graph = initialize_graph(graph, cases_list)
        for case in cases_list:
            trace_distance, time_distance = extract_case_distances(graph, case)
            case.graph_distance = trace_distance
            case.time_distance = time_distance

        denstream.dbscan(cases_list)
        assert len(denstream.p_micro_clusters) == 1
        assert denstream.all_cases.keys() == {'1', '4'}
        new_p_mc = denstream.p_micro_clusters[0]
        assert new_p_mc.weight == 2
        assert new_p_mc.creation_time == 0
        assert new_p_mc.lambda_ == 0.15

    def test_train(self, cases_list):
        anomalous_denstream = DenStream(2, 0.15, 0.3, 0.1, 4, 1000)
        case_1 = cases_list[0]
        anomalous_denstream.train(case_1)
        assert anomalous_denstream.all_cases == {'1': 0}
        assert anomalous_denstream.time == 0
        assert anomalous_denstream.no_processed_points == 1
        assert anomalous_denstream.p_micro_clusters == []
        assert not anomalous_denstream.o_micro_clusters == []
        assert anomalous_denstream.o_micro_clusters[0].id == 0
        assert np.isnan(sum(anomalous_denstream.o_micro_clusters[0].CF))
        assert np.isnan(sum(anomalous_denstream.o_micro_clusters[0].CF))
        assert anomalous_denstream.o_micro_clusters[0].weight == 1
        assert anomalous_denstream.o_micro_clusters[0].lambda_ == 0.15
        assert anomalous_denstream.o_micro_clusters[0].creation_time == 0

        denstream = DenStream(2, 0.15, 0.3, 0.1, 4, 1000)
        graph = nx.DiGraph()
        graph = initialize_graph(graph, cases_list)
        for case in cases_list:
            trace_distance, time_distance = extract_case_distances(graph, case)
            case.graph_distance = trace_distance
            case.time_distance = time_distance

        denstream.dbscan(cases_list)

        assert len(denstream.p_micro_clusters) == 1
        assert len(denstream.o_micro_clusters) == 0
        assert denstream.all_cases == {'1': 0, '4': 0}
        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].CF == [0.25, 0])
        assert np.all(denstream.p_micro_clusters[0].CF2 == [0.03125, 0])
        assert denstream.p_micro_clusters[0].weight == 2

        denstream.train(cases_list[2])
        assert denstream.all_cases
        assert denstream.all_cases['3'] == 0
        assert len(denstream.o_micro_clusters) == 0
        assert len(denstream.p_micro_clusters) == 1
        assert np.all(denstream.p_micro_clusters[0].CF == [(0.25 + (1 / 3)), 0])
        assert np.all(denstream.p_micro_clusters[0].CF2 == [0.03125 + (1 / 3) * (1 / 3), 0])
        assert denstream.p_micro_clusters[0].weight == 3
        assert denstream.no_processed_points == 1

        # self.no_processed_points % self.stream_speed == 0
        denstream = DenStream(2, 0.15, 0.3, 0.1, 4, 1000)
        graph = nx.DiGraph()
        graph = initialize_graph(graph, cases_list)
        for case in cases_list:
            trace_distance, time_distance = extract_case_distances(graph, case)
            case.graph_distance = trace_distance
            case.time_distance = time_distance

        denstream.dbscan(cases_list)

        assert len(denstream.p_micro_clusters) == 1
        assert len(denstream.o_micro_clusters) == 0
        assert denstream.all_cases == {'1': 0, '4': 0}
        assert denstream.p_micro_clusters[0].id == 0
        assert np.all(denstream.p_micro_clusters[0].CF == [0.25, 0])
        assert np.all(denstream.p_micro_clusters[0].CF2 == [0.03125, 0])
        assert denstream.p_micro_clusters[0].weight == 2

        denstream.stream_speed = 1
        denstream.train(cases_list[2])
        assert denstream.all_cases
        assert denstream.all_cases['3'] == 0
        assert len(denstream.o_micro_clusters) == 0
        assert len(denstream.p_micro_clusters) == 1
        # assert np.all(denstream.p_micro_clusters[0].CF == [(0.25 * (2 ** (-0.15))), 0])
        # assert np.all(denstream.p_micro_clusters[0].CF2 == [0.03125 * (2 ** (-0.15)), 0])

        # mc.weight > self.beta * self.mu
        assert denstream.p_micro_clusters[0].weight == 3 * (2 ** (-0.15))
        assert denstream.no_processed_points == 1
        # assert denstream.tp == 18 ... true???
        # assert ceil(0.3877443751) == 1
        # assert denstream.tp == ceil(0.3877443751)
        assert denstream.time == 1

    def test_generate_clusters(self, denstream):
        assert len(denstream.p_micro_clusters) == 0

        dense_group, not_dense_group = denstream.generate_clusters()
        assert dense_group == [[]]
        assert not_dense_group == [[]]

        case_0 = Case('0')
        case_1 = Case('1')

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([5.0, 5.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
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
        assert cluster.case_ids == ['0', '1']

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([3.0, 3.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
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
        assert cluster.case_ids == ['0', '1']

        case_2 = Case('2')
        case_3 = Case('3')

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([3.0, 3.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([4.0, 4.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) <= 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        cluster = cluster_list[1]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [1, 1])
        assert cluster.radius == 0
        assert cluster.weight == 4
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert dense_group[1]
            assert cluster_list[2]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([3.0, 3.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 1
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([3.0, 3.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 1
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) <= 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        cluster = cluster_list[1]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [3, 3])
        assert cluster.radius == 0
        assert cluster.weight == 1
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert cluster_list[2]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([3.0, 3.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 1.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) > 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.25, 0.25])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 4
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert dense_group[1]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([6.0, 6.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 6
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 1.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) > 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = not_dense_group[0]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.5, 0.5])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 2
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert not_dense_group[1]
            assert dense_group[1]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([6.0, 6.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 6
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 1.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 4
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) > 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = dense_group[1]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.25, 0.25])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 4
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert dense_group[2]
            assert cluster_list[1]

        micro_cluster = MicroCluster(0, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 1.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 3
        denstream.p_micro_clusters = []
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_0.id] = micro_cluster.id
        denstream.all_cases[case_1.id] = micro_cluster.id

        micro_cluster = MicroCluster(1, 2, 0, 0.15)
        micro_cluster.CF = np.array([1.0, 1.0])
        micro_cluster.CF2 = np.array([1.0, 1.0])
        micro_cluster.weight = 2
        denstream.p_micro_clusters.append(micro_cluster)
        denstream.all_cases[case_2.id] = micro_cluster.id
        denstream.all_cases[case_3.id] = micro_cluster.id

        cl1 = denstream.p_micro_clusters[0]
        cl2 = denstream.p_micro_clusters[1]

        assert len(denstream.p_micro_clusters) > 1
        assert denstream.euclidean_distance(cl1.centroid, cl2.centroid) > 2 * denstream.epsilon
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
        assert cluster.case_ids == ['0', '1']
        with pytest.raises(IndexError):
            assert cluster_list[1]
        cluster_list = not_dense_group[1]
        cluster = cluster_list[0]
        assert isinstance(cluster, Cluster)
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0.5, 0.5])
        assert cluster.radius == cl2.radius
        assert cluster.weight == 2
        assert cluster.case_ids == ['2', '3']
        with pytest.raises(IndexError):
            assert not_dense_group[2]
            assert cluster_list[1]

    def test_generate_outlier_clusters(self, denstream):
        assert denstream.generate_outlier_clusters() == []

        case_0 = Case('0')
        case_1 = Case('1')
        case_2 = Case('2')
        case_3 = Case('3')

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

        assert denstream.all_cases == {'0': 0, '1': 1, '2': 3, '3': 3}

        cluster_list = denstream.generate_outlier_clusters()

        cluster = cluster_list[0]
        assert cluster.id == 0
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 3
        assert cluster.case_ids == ['0']

        cluster = cluster_list[1]
        assert cluster.id == 1
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 4
        assert cluster.case_ids == ['1']

        cluster = cluster_list[2]
        assert cluster.id == 2
        assert np.all(cluster.centroid == [0, 0])
        assert cluster.radius == 0
        assert cluster.weight == 6
        assert cluster.case_ids == ['2', '3']
