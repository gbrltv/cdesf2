from cdesf2.core import Process
from cdesf2.clustering import DenStream
from cdesf2.data_structures import Case
from datetime import datetime
from math import log10
from os import makedirs, path, remove, rmdir
import shutil
import networkx as nx
import numpy as np
import json
from cdesf2.utils import initialize_graph,\
    normalize_graph, extract_case_distances
import pytest


class TestProcess:
    @pytest.fixture
    def denstream_kwargs(self):
        denstream_kwargs = {'n_features': 2,
                            'beta': 0.3,
                            'lambda_': 0.15,
                            'epsilon': 0.1,
                            'mu': 4,
                            'stream_speed': 1000}
        return denstream_kwargs

    @pytest.fixture
    def process(self, denstream_kwargs):
        process = Process('test', datetime(2015, 5, 10, 8, 22, 53),
                          43200, False, 'plot_path', False, 'metric_path',
                          './visualization', denstream_kwargs)
        return process

    @pytest.fixture
    def cases_list(self):
        case_1 = Case('1')
        case_1.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_1.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_1.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))

        case_2 = Case('2')
        case_2.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_2.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))

        case_3 = Case('3')
        case_3.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case_3.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case_3.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_3.set_activity('activityD', datetime(2015, 5, 10, 8, 00, 30))

        return [case_3, case_2, case_1]

    def test_initial_value(self, process, denstream_kwargs):
        assert isinstance(process.name, str)
        assert process.name == 'test'
        assert isinstance(process.gen_plot, bool)
        assert not process.gen_plot
        assert isinstance(process.plot_path, str)
        assert process.plot_path == 'plot_path'
        assert isinstance(process.gen_metrics, bool)
        assert not process.gen_metrics
        assert isinstance(process.metrics_path, str)
        assert process.metrics_path == 'metric_path'
        assert process.event_count == 0
        assert process.total_cases == set()
        assert process.cases == []
        assert isinstance(process.time_horizon, int)
        assert process.time_horizon == 43200
        # assert process.act_dic == {}
        assert not process.initialized
        assert isinstance(process.check_point, datetime)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert process.cp_count == 0
        assert process.nyquist == 0
        assert process.check_point_cases == 0

        # self.process_model_graph = nx.DiGraph()
        assert isinstance(process.process_model_graph, nx.DiGraph)
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

        # self.denstream == DenStream(**denstream_kwargs)
        assert isinstance(process.denstream, DenStream)
        assert process.denstream.n_features == 2
        assert process.denstream.lambda_ == 0.15
        assert process.denstream.beta == 0.3
        assert process.denstream.epsilon == 0.1
        assert process.denstream.mu == 4
        assert process.denstream.p_micro_clusters == []
        assert process.denstream.o_micro_clusters == []
        assert process.denstream.time == 0
        assert process.denstream.all_cases == {}
        assert process.denstream.stream_speed == 1000

        assert process.cluster_metrics == []
        assert process.case_metrics == []
        # assert process.pmg_by_cp == []

    def test_no_value(self):
        with pytest.raises(Exception):
            assert Process()

    def test_initialize_case_metrics(self, process, cases_list):

        process.initialize_case_metrics()
        assert process.cases == []

        process.cases = cases_list

        case_3 = process.cases[0]
        assert np.isnan(case_3.graph_distance)
        assert np.isnan(case_3.time_distance)
        case_2 = process.cases[1]
        assert np.isnan(case_2.graph_distance)
        assert np.isnan(case_2.time_distance)
        case_1 = process.cases[2]
        assert np.isnan(case_1.graph_distance)
        assert np.isnan(case_1.time_distance)

        process.initialize_case_metrics()

        case_3 = process.cases[0]
        assert case_3.graph_distance == 0
        assert case_3.time_distance == 0
        case_2 = process.cases[1]
        assert case_2.graph_distance == 0
        assert case_2.time_distance == 0
        case_1 = process.cases[2]
        assert case_1.graph_distance == 0
        assert case_1.time_distance == 0

        process.cases = cases_list
        process.process_model_graph = initialize_graph(nx.DiGraph(), process.cases)
        process.initialize_case_metrics()
        case_3 = process.cases[0]

        graph_dist_3, time_dist_3 = extract_case_distances(process.process_model_graph, case_3)
        assert case_3.graph_distance == graph_dist_3
        assert case_3.time_distance == time_dist_3

        graph_dist_2, time_dist_2 = extract_case_distances(process.process_model_graph, case_2)
        case_2 = process.cases[1]
        assert case_2.graph_distance == graph_dist_2
        assert case_2.time_distance == time_dist_2

        graph_dist_1, time_dist_1 = extract_case_distances(process.process_model_graph, case_1)
        case_1 = process.cases[2]
        assert case_1.graph_distance == graph_dist_1
        assert case_1.time_distance == time_dist_1

    def test_get_case(self, process):
        case1 = Case('1')
        case2 = Case('2')
        case3 = Case('3')
        case4 = Case('4')
        process.cases = [case1, case2, case3, case4]

        case = process.get_case('1')
        assert case == 0
        case = process.get_case('3')
        assert case == 2

        process.cases = [case4, case1, case3, case2]
        case = process.get_case('1')
        assert case == 1
        case = process.get_case('2')
        assert case == 3

        case = process.get_case('5')
        assert case is None

    def test_release_cases_from_memory(self, process):
        case1 = Case('1')
        case2 = Case('2')
        case3 = Case('3')
        case4 = Case('4')
        case4.set_activity('activity1', datetime(2015, 5, 10, 8, 30, 00))
        case1.set_activity('activity1', datetime(2015, 5, 10, 9, 00, 00))
        case3.set_activity('activity1', datetime(2015, 5, 10, 9, 30, 00))
        case2.set_activity('activity1', datetime(2015, 5, 10, 10, 00, 00))

        process.cases = [case2, case3, case1, case4]
        process.release_cases_from_memory()
        assert process.cases == []

        process.cases = [case2, case3, case1, case4]
        process.nyquist = 1
        process.release_cases_from_memory()
        assert process.cases == [case2]

        process.cases = [case2, case3, case1, case4]
        process.nyquist = 2
        process.release_cases_from_memory()
        assert process.cases == [case2, case3]

        process.cases = [case2, case3, case1, case4]
        process.nyquist = 3
        process.release_cases_from_memory()
        assert process.cases == [case2, case3, case1]

        process.cases = [case2, case3, case1, case4]
        process.nyquist = 4
        process.release_cases_from_memory()
        assert process.cases == [case2, case3, case1, case4]

    def test_initialize_cdesf(self, process, cases_list):
        assert process.nyquist == 0
        assert process.check_point_cases == 0

        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

        assert process.cases == []
        with pytest.raises(Exception):
            assert initialize_graph(nx.DiGraph(), [])

        case4 = Case('4')
        case4.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case4.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 10))
        case4.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))

        process.cases = cases_list
        process.cases.insert(0, case4)
        process.process_model_graph = initialize_graph(nx.DiGraph(), process.cases)
        pmg = initialize_graph(nx.DiGraph(), cases_list)
        for case in cases_list:
            graph_distance, time_distance = extract_case_distances(pmg, case)
            case.graph_distance = graph_distance
            case.time_distance = time_distance
        case_4 = process.cases[0]

        process.initialize_cdesf()

        graph_dist_4, time_dist_4 = extract_case_distances(pmg, case_4)
        assert case_4.graph_distance == graph_dist_4
        assert case_4.time_distance == time_dist_4

        assert len(process.denstream.p_micro_clusters) == 1
        assert process.denstream.all_cases.keys() == {'1', '4'}
        new_p_mc = process.denstream.p_micro_clusters[0]
        assert new_p_mc.weight == 2
        assert new_p_mc.creation_time == 0
        assert new_p_mc.lambda_ == 0.15

        assert process.initialized

    def test_set_case(self, process):
        assert process.check_point_cases == 0
        assert process.cases == []
        process.set_case('5', 'activityA', datetime(2015, 5, 10, 8, 00, 00))
        assert process.check_point_cases == 1
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == '5'
        activity = case_test.activities[0]
        assert activity.name == 'activityA'
        assert activity.timestamp == datetime(2015, 5, 10, 8, 00, 00)

        case1 = Case('1')
        case2 = Case('2')
        case3 = Case('3')
        case4 = Case('4')
        process.check_point_cases = 0
        process.cases = [case1, case2, case3, case4]
        process.set_case('3', 'activityA', datetime(2015, 5, 10, 8, 00, 00))
        assert process.check_point_cases == 0
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == '3'
        activity = case_test.activities[0]
        assert activity.name == 'activityA'
        assert activity.timestamp == datetime(2015, 5, 10, 8, 00, 00)

    def test_check_point_update(self, process, cases_list):

        # self.cases = nyquist (0=0)
        assert process.cp_count == 0

        process.check_point_update()

        assert process.cp_count == 1
        assert process.check_point_cases == 0
        assert process.nyquist == 0
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

        # self.cases < nyquist
        process.nyquist = 1

        process.check_point_update()

        assert process.cp_count == 2
        assert process.nyquist == 1
        assert process.check_point_cases == 0
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

        # self.cases > nyquist
        # nyquist = 0
        # process_model_graph = 0
        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 0

        with pytest.raises(Exception):
            assert process.check_point_update()
        # initialize_graph -> extract_cases_time_and_trace
        # Exception("Extracting trace and timestamp list out of a list with no cases")
        assert process.cp_count == 3  # even if error, cp_count incremented!!!

        # self.cases > nyquist
        # nyquist = 1
        # process_model_graph = 0
        case3 = cases_list[0]
        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 1

        process.check_point_update()

        assert process.cp_count == 4
        assert process.cases == [case3]
        assert process.check_point_cases == 0
        assert process.nyquist == 1 * 2
        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        check_point_graph = initialize_graph(nx.DiGraph(), process.cases)
        for node1, node2, data in process.process_model_graph.edges.data():
            assert process.process_model_graph[node1][node2] == check_point_graph[node1][node2]
            assert process.process_model_graph[node1][node2]['weight'] == check_point_graph[node1][node2]['weight']
            assert process.process_model_graph[node1][node2]['time'] == check_point_graph[node1][node2]['time']

        # self.cases > nyquist
        # nyquist = 1
        # process_model_graph = 1
        process.process_model_graph = initialize_graph(nx.DiGraph(), [cases_list[2]])
        pmg = process.process_model_graph
        assert pmg['activityA']['activityB']
        assert pmg['activityB']['activityC']
        with pytest.raises(Exception):
            assert pmg['activityC']['activityD']
            assert pmg['activityD']['activityE']
        case3 = cases_list[0]
        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 1

        process.check_point_update()

        assert process.cp_count == 5
        assert process.cases == [case3]
        assert process.check_point_cases == 0
        assert process.nyquist == 1 * 2
        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        assert pmg['activityA']['activityB']
        assert pmg['activityB']['activityC']
        assert pmg['activityC']['activityD']
        check_point_graph = initialize_graph(nx.DiGraph(), process.cases)
        for node1, node2, data in process.process_model_graph.edges.data():
            pmg[node1][node2]['weight'] *= 0.95
            pmg[node1][node2]['weight'] += check_point_graph[node1][node2]['weight']
            pmg[node1][node2]['time'] += check_point_graph[node1][node2]['time']
            pmg = normalize_graph(pmg)
            assert process.process_model_graph[node1][node2]['weight'] == pmg[node1][node2]['weight']
            assert process.process_model_graph[node1][node2]['weight_normalized'] == \
                pmg[node1][node2]['weight_normalized']
            assert process.process_model_graph[node1][node2]['time'] == pmg[node1][node2]['time']
            assert process.process_model_graph[node1][node2]['time_normalized'] == \
                pmg[node1][node2]['time_normalized']

        # self.cases > nyquist
        # nyquist = 2
        # process_model_graph = 1
        case2 = cases_list[1]
        case3 = cases_list[0]

        process.process_model_graph = initialize_graph(nx.DiGraph(), [cases_list[1]])
        pmg = process.process_model_graph
        assert pmg['activityA']['activityB']
        with pytest.raises(Exception):
            assert pmg['activityC']['activityD']
            assert pmg['activityD']['activityE']

        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 2

        process.check_point_update()

        assert process.cp_count == 6
        assert process.cases == [case3, case2]
        assert process.check_point_cases == 0
        assert process.nyquist == 1 * 2

        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        assert process.process_model_graph['activityA']['activityB']
        assert process.process_model_graph['activityB']['activityC']
        assert process.process_model_graph['activityC']['activityD']
        check_point_graph = initialize_graph(nx.DiGraph(), process.cases)
        for node1, node2, data in process.process_model_graph.edges.data():
            pmg[node1][node2]['weight'] *= 0.95
            pmg[node1][node2]['weight'] += check_point_graph[node1][node2]['weight']
            pmg[node1][node2]['time'] += check_point_graph[node1][node2]['time']
            pmg = normalize_graph(pmg)
            assert process.process_model_graph[node1][node2]['weight'] == pmg[node1][node2]['weight']
            assert process.process_model_graph[node1][node2]['weight_normalized'] == \
                pmg[node1][node2]['weight_normalized']
            assert process.process_model_graph[node1][node2]['time'] == pmg[node1][node2]['time']
            assert process.process_model_graph[node1][node2]['time_normalized'] == \
                pmg[node1][node2]['time_normalized']

        # self.cases > nyquist
        # nyquist = 2
        # process_model_graph = 2
        case2 = cases_list[1]
        case3 = cases_list[0]

        process.process_model_graph = initialize_graph(nx.DiGraph(), [cases_list[1], cases_list[2]])
        pmg = process.process_model_graph
        assert pmg['activityA']['activityB']
        assert pmg['activityB']['activityC']
        with pytest.raises(Exception):
            assert pmg['activityC']['activityD']
            assert pmg['activityD']['activityE']

        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 2

        process.check_point_update()

        assert process.cp_count == 7
        assert process.cases == [case3, case2]
        assert process.check_point_cases == 0
        assert process.nyquist == 1 * 2

        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        assert process.process_model_graph['activityA']['activityB']
        assert process.process_model_graph['activityB']['activityC']
        assert process.process_model_graph['activityC']['activityD']
        check_point_graph = initialize_graph(nx.DiGraph(), process.cases)
        for node1, node2, data in process.process_model_graph.edges.data():
            pmg[node1][node2]['weight'] *= 0.95
            pmg[node1][node2]['weight'] += check_point_graph[node1][node2]['weight']
            pmg[node1][node2]['time'] += check_point_graph[node1][node2]['time']
            pmg = normalize_graph(pmg)
            assert process.process_model_graph[node1][node2]['weight'] == pmg[node1][node2]['weight']
            assert process.process_model_graph[node1][node2]['weight_normalized'] == \
                pmg[node1][node2]['weight_normalized']
            assert process.process_model_graph[node1][node2]['time'] == pmg[node1][node2]['time']
            assert process.process_model_graph[node1][node2]['time_normalized'] == \
                pmg[node1][node2]['time_normalized']

    def test_process_event(self, process):

        process.gen_metrics = True
        assert process.event_count == 0
        assert process.cases == []
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)

        process.process_event('1', 'activityA', datetime(2015, 5, 10, 8, 00, 00), 0)
        assert process.event_count == 1
        assert process.total_cases == {'1'}

        assert process.check_point_cases == 1
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == '1'
        activity = case_test.activities[0]
        assert activity.name == 'activityA'
        assert activity.timestamp == datetime(2015, 5, 10, 8, 00, 00)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert not process.initialized

        process.process_event('2', 'activityA', datetime(2015, 5, 10, 8, 00, 00), 0)
        process.process_event('1', 'activityB', datetime(2015, 5, 10, 8, 00, 10), 0)
        process.process_event('3', 'activityA', datetime(2015, 5, 10, 8, 00, 00), 0)
        process.process_event('3', 'activityB', datetime(2015, 5, 10, 8, 00, 10), 0)
        process.process_event('4', 'activityC', datetime(2015, 5, 10, 8, 00, 20), 0)
        process.process_event('3', 'activityC', datetime(2015, 5, 10, 8, 00, 20), 0)
        process.process_event('4', 'activityD', datetime(2015, 5, 10, 8, 00, 30), 0)
        process.process_event('5', 'activityA', datetime(2015, 5, 10, 8, 00, 00), 0)
        process.process_event('5', 'activityD', datetime(2015, 5, 10, 8, 00, 30), 0)

        assert process.check_point_cases == 5
        assert process.total_cases == {'1', '2', '3', '4', '5'}
        assert process.cases
        assert len(process.cases) == 5
        case_test = process.cases[0]
        assert case_test.id == '5'
        activity = case_test.activities[0]
        assert activity.name == 'activityA'
        assert activity.timestamp == datetime(2015, 5, 10, 8, 00, 00)
        activity = case_test.activities[1]
        assert activity.name == 'activityD'
        assert activity.timestamp == datetime(2015, 5, 10, 8, 00, 30)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert not process.initialized

        process.process_event('4', 'activityE', datetime(2015, 5, 10, 8, 00, 40), 0)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert process.check_point_cases == 5
        assert not process.initialized

        process.process_event('4', 'activityE', datetime(2015, 5, 10, 21, 00, 40), 0)
        assert process.initialized
        assert process.check_point_cases == 5
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        #print(list(process.process_model_graph.edges(data=True)))
        assert len(process.process_model_graph.edges) == 6
        assert len(process.denstream.p_micro_clusters) == 2
        assert len(process.denstream.o_micro_clusters) == 0
        assert process.nyquist == 10

        process.process_event('1', 'activityC', datetime(2015, 5, 10, 21, 00, 50), 0)
        process.process_event('1', 'activityD', datetime(2015, 5, 10, 21, 50, 50), 0)
        assert process.check_point_cases == 5
        assert process.total_cases == {'1', '2', '3', '4', '5'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5'}
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        assert len(process.denstream.p_micro_clusters) == 2
        assert len(process.denstream.o_micro_clusters) == 2
        assert process.nyquist == 10

        process.process_event('1', 'activityE', datetime(2015, 5, 10, 21, 50, 50), 0)
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert process.check_point_cases == 5
        assert process.total_cases == {'1', '2', '3', '4', '5'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5'}
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        assert len(process.denstream.p_micro_clusters) == 3
        assert len(process.denstream.o_micro_clusters) == 1
        assert process.nyquist == 10

        process.process_event('6', 'activityA', datetime(2015, 5, 11, 10, 50, 50), 0)
        process.process_event('7', 'activityA', datetime(2015, 5, 11, 10, 50, 50), 0)
        process.process_event('8', 'activityA', datetime(2015, 5, 11, 10, 50, 50), 0)
        process.process_event('9', 'activityA', datetime(2015, 5, 11, 10, 50, 50), 0)
        process.process_event('10', 'activityA', datetime(2015, 5, 11, 10, 50, 50), 0)
        process.process_event('2', 'activityB', datetime(2015, 5, 12, 22, 50, 50), 0)
        process.process_event('2', 'activityB', datetime(2015, 5, 13, 22, 50, 50), 0)

        assert process.cp_count == 1
        assert path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        remove(f'./visualization/process_model_graph_{process.cp_count}.json')

        assert process.check_point_cases == 0
        assert process.check_point == datetime(2015, 5, 13, 22, 50, 50)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        assert len(process.denstream.p_micro_clusters) == 4
        assert len(process.denstream.o_micro_clusters) == 1
        assert process.total_cases == {'1', '2', '3', '4', '5',
                                       '6', '7', '8', '9', '10'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5'}
        assert process.nyquist == 10
        assert len(process.cases) == 10

        process.process_event('11', 'activityB', datetime(2015, 5, 13, 23, 00, 00), 0)
        assert len(process.cases) == 11
        assert process.total_cases == {'1', '2', '3', '4', '5',
                                       '6', '7', '8', '9', '10', '11'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5'}
        assert process.nyquist == 10
        process.process_event('11', 'activityC', datetime(2015, 5, 14, 12, 00, 00), 0)

        assert process.cp_count == 2
        assert path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        remove(f'./visualization/process_model_graph_{process.cp_count}.json')
        rmdir('./visualization')

        assert len(process.cases) == 10
        assert process.total_cases == {'1', '2', '3', '4', '5',
                                       '6', '7', '8', '9', '10', '11'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5', '11'}
        assert process.nyquist == 2

        process.process_event('11', 'activityE', datetime(2015, 5, 15, 1, 00, 00), 0)
        assert process.cp_count == 3
        assert path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        remove(f'./visualization/process_model_graph_{process.cp_count}.json')
        rmdir('./visualization')

        assert len(process.cases) == 2
        assert process.cases[0].id == '11'
        assert process.cases[1].id == '2'
        assert process.total_cases == {'1', '2', '3', '4', '5',
                                       '6', '7', '8', '9', '10', '11'}
        assert process.denstream.all_cases.keys() == {'1', '2', '3', '4', '5', '11'}
        assert process.nyquist == 0

        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 8
        assert len(process.denstream.p_micro_clusters) == 4
        assert len(process.denstream.o_micro_clusters) == 3

    def test_gen_case_metrics(self, process):

        case1 = Case('1')
        case1.graph_distance = 0.5
        case1.time_distance = 1.5
        case2 = Case('2')
        case2.graph_distance = 1.0
        case2.time_distance = 2.0
        case3 = Case('3')
        case3.graph_distance = 0.25
        case3.time_distance = 1.25
        case4 = Case('4')
        case4.graph_distance = 0.75
        case4.time_distance = 1.75

        process.cases = [case1, case2, case3, case4]

        assert not process.case_metrics
        process.gen_case_metrics(1, 0)
        case_metrics = process.case_metrics[0]
        assert case_metrics[0] == 1
        assert case_metrics[1] == '1'
        assert case_metrics[2] == 0.5
        assert case_metrics[3] == 1.5
        with pytest.raises(Exception):
            assert process.case_metrics[1]
            assert process.case_metrics[2]
            assert process.case_metrics[3]

        process.gen_case_metrics(2, 1)
        assert process.case_metrics[0]
        case_metrics = process.case_metrics[1]
        assert case_metrics[0] == 2
        assert case_metrics[1] == '2'
        assert case_metrics[2] == 1.0
        assert case_metrics[3] == 2.0
        with pytest.raises(Exception):
            assert process.case_metrics[2]
            assert process.case_metrics[3]

        process.gen_case_metrics(3, 2)
        assert process.case_metrics[0]
        assert process.case_metrics[1]
        case_metrics = process.case_metrics[2]
        assert case_metrics[0] == 3
        assert case_metrics[1] == '3'
        assert case_metrics[2] == 0.25
        assert case_metrics[3] == 1.25
        with pytest.raises(Exception):
            assert process.case_metrics[3]

        process.gen_case_metrics(4, 3)
        assert process.case_metrics[0]
        assert process.case_metrics[1]
        assert process.case_metrics[2]
        case_metrics = process.case_metrics[3]
        assert case_metrics[0] == 4
        assert case_metrics[1] == '4'
        assert case_metrics[2] == 0.75
        assert case_metrics[3] == 1.75

    def test_save_pmg_on_check_point(self, process, cases_list):
        assert process.path_to_json == './visualization'

        assert len(process.process_model_graph.edges) == 0
        process.save_pmg_on_check_point()
        assert process.cp_count == 0
        assert path.exists('./visualization')
        assert not path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')

        process.cases = cases_list
        process.process_model_graph = initialize_graph(nx.DiGraph(), process.cases)
        pmg = initialize_graph(nx.DiGraph(), process.cases)
        pmg = normalize_graph(pmg)
        process.initialize_case_metrics()
        process.save_pmg_on_check_point()

        assert process.cp_count == 0
        assert path.exists('./visualization')
        assert path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        invalid_cp_count = process.cp_count + 1
        assert not path.isfile(f'./visualization/process_model_graph_{invalid_cp_count}.json')
        with open(f'./visualization/process_model_graph_{process.cp_count}.json') as file:
            data = json.load(file)

        nodes = data['nodes']
        edges = data['links']
        activities_ids = [node['id'] for node in nodes]
        assert len(edges) == len(pmg.edges)
        assert activities_ids == list(pmg)
        i = 0
        for node1, node2, data in pmg.edges(data=True):
            edge = edges[i]
            edge_nodes = (edge['source'], edge['target'])
            assert edge_nodes == (node1, node2)
            assert edge['weight'] == data['weight']
            assert edge['time'] == data['time']
            assert edge['weight_normalized'] == data['weight_normalized']
            assert edge['time_normalized'] == data['time_normalized']
            i += 1

        remove(f'./visualization/process_model_graph_{process.cp_count}.json')

        process.cp_count = 1
        assert not path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')

        pmg = normalize_graph(pmg)
        process.save_pmg_on_check_point()

        assert path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        with open(f'./visualization/process_model_graph_{process.cp_count}.json') as file:
            data = json.load(file)

        nodes = data['nodes']
        edges = data['links']
        activities_ids = [d['id'] for d in nodes]
        assert len(edges) == len(pmg.edges)
        assert activities_ids == list(pmg)
        i = 0
        for node1, node2, data in pmg.edges(data=True):
            edge = edges[i]
            edge_nodes = (edge['source'], edge['target'])
            assert edge_nodes == (node1, node2)
            assert edge['weight'] == data['weight']
            assert edge['time'] == data['time']
            assert edge['weight_normalized'] == data['weight_normalized']
            assert edge['time_normalized'] == data['time_normalized']
            i += 1

        remove(f'./visualization/process_model_graph_{process.cp_count}.json')
        rmdir('./visualization')

    def test_run_cdesf(self, process):
        dir_path = 'demo'
        filename = 'Detail_Supplier_IW-Frozen.csv'
        process.run_cdesf(dir_path, filename)

        assert not process.initialized
        assert not process.check_point_cases == 0
        assert len(process.process_model_graph.nodes) == 0
        assert len(process.process_model_graph.edges) == 0
        assert not path.isfile(f'./visualization/process_model_graph_{process.cp_count}.json')
        #shutil.rmtree('./visualization')