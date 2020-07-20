from cdesf2.core import Process
from cdesf2.clustering import DenStream
from cdesf2.data_structures import Case
from datetime import datetime
from math import log10
import networkx as nx
import numpy as np
from cdesf2.utils import initialize_graph, normalize_graph
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
                          43200, False, 'plot_path', False, 'metric_path', denstream_kwargs)
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

        return [case_1, case_2, case_3]

    def test_initial_value(self, process, denstream_kwargs):
        assert type(process.name) is str
        assert process.name == 'test'
        assert type(process.gen_plot) is bool
        assert not process.gen_plot
        assert type(process.plot_path) is str
        assert process.plot_path == 'plot_path'
        assert type(process.gen_metrics) is bool
        assert not process.gen_metrics
        assert type(process.metrics_path) is str
        assert process.metrics_path == 'metric_path'
        assert process.event_count == 0
        assert process.total_cases == set()
        assert process.cases == []
        assert type(process.time_horizon) is int
        assert process.time_horizon == 43200
        assert process.act_dic == {}
        assert not process.initialized
        assert type(process.check_point) is datetime
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert process.cp_count == 0
        assert process.nyquist == 0
        assert process.check_point_cases == 0

        # self.process_model_graph = nx.DiGraph()
        assert type(process.process_model_graph) is nx.DiGraph
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

        # self.denstream == DenStream(**denstream_kwargs)
        assert type(process.denstream) is DenStream
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
        assert process.pmg_by_cp == []

    def test_no_value(self):
        with pytest.raises(Exception):
            assert Process()

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

        process.cases = [case1, case2, case3, case4]
        process.release_cases_from_memory()
        assert process.cases == []

        process.cases = [case1, case2, case3, case4]
        process.nyquist = 1
        process.release_cases_from_memory()
        assert process.cases == [case2]

        process.cases = [case1, case2, case3, case4]
        process.nyquist = 4
        process.release_cases_from_memory()
        assert process.cases == [case2, case3, case1, case4]

    # def test_initialize_case_metrics(self, process, cases_list):
    #     process.cases = cases_list
    #     assert cases_list[0].graph_distance == 0.25
    #     assert cases_list[0].time_distance == 0
    #
    #     assert cases_list[1].graph_distance == 0
    #     assert cases_list[1].time_distance == 0
    #
    #     assert cases_list[2].graph_distance == 0.5
    #     assert cases_list[2].time_distance == log10(10 / 20)

    # def test_gen_case_metrics(self, process, cases_list):
    #     process.cases = cases_list
    #
    #     assert not process.case_metrics
    #     process.gen_case_metrics(1, 0)
    #     case_metrics = process.case_metrics[0]
    #     assert case_metrics[0] == 1
    #     assert case_metrics[1] == '1'
    #     assert case_metrics[2] == 0.25
    #     assert case_metrics[3] == 0
    #     with pytest.raises(Exception):
    #         assert process.case_metrics[1]
    #         assert process.case_metrics[2]
    #
    #     process.gen_case_metrics(2, 1)
    #     assert process.case_metrics[0]
    #     case_metrics = process.case_metrics[1]
    #     assert case_metrics[0] == 2
    #     assert case_metrics[1] == '2'
    #     assert case_metrics[2] == 0
    #     assert case_metrics[3] == 0
    #     with pytest.raises(Exception):
    #         assert process.case_metrics[2]
    #
    #     process.gen_case_metrics(3, 2)
    #     assert process.case_metrics[0]
    #     assert process.case_metrics[1]
    #     case_metrics = process.case_metrics[2]
    #     assert case_metrics[0] == 3
    #     assert case_metrics[1] == '3'
    #     assert case_metrics[2] == 0.5
    #     assert case_metrics[3] == log10(10 / 20)

    # def test_gen_pmg_metrics(self, process, cases_list):
    #     process.cases = cases_list
    #     process.process_model_graph = initialize_graph(nx.DiGraph(), process.cases)
    #     with pytest.raises(Exception):
    #         assert process.pmg_by_cp[0]
    #         assert process.pmg_by_cp[1]
    #         assert process.pmg_by_cp[2]
    #
    #     process.gen_pmg_metrics()
    #
    #     pmg_by_cp = process.pmg_by_cp[0]
    #     assert pmg_by_cp[0] == 0
    #     assert pmg_by_cp[1] == ('activityA', 'activityB')
    #     assert pmg_by_cp[2] == 3
    #     assert pmg_by_cp[3] == 30.0
    #     assert pmg_by_cp[4] == 1.0
    #     assert pmg_by_cp[5] == 10.0
    #
    #     pmg_by_cp = process.pmg_by_cp[1]
    #     assert pmg_by_cp[0] == 0
    #     assert pmg_by_cp[1] == ('activityB', 'activityC')
    #     assert pmg_by_cp[2] == 2
    #     assert pmg_by_cp[3] == 20.0
    #     assert pmg_by_cp[4] == 2 / 3
    #     assert pmg_by_cp[5] == 10.0
    #
    #     pmg_by_cp = process.pmg_by_cp[2]
    #     assert pmg_by_cp[0] == 0
    #     assert pmg_by_cp[1] == ('activityC', 'activityD')
    #     assert pmg_by_cp[2] == 1
    #     assert pmg_by_cp[3] == 10.0
    #     assert pmg_by_cp[4] == 1 / 3
    #     assert pmg_by_cp[5] == 10.0

    # def test_initialize_cdesf(self, process, cases_list):
    #     current_time = datetime(2015, 5, 10, 8, 00, 30)
    #     process.cases = cases_list
    #     process.gen_metrics = True
    #     pmg = initialize_graph(nx.DiGraph(), process.cases)
    #     if process.gen_metrics:
    #         pmg = normalize_graph(pmg)
    #     process.initialize_cdesf(current_time)
    #     assert process.nyquist == 0
    #     assert process.check_point == current_time
    #
    #     assert len(process.process_model_graph.edges) == len(pmg.edges)
    #     assert len(process.process_model_graph.nodes) == len(pmg.nodes)
    #     for node1, node2, data in (process.process_model_graph.edges(data=True)):
    #         assert process.process_model_graph[node1][node2] == pmg[node1][node2]
    #         assert process.process_model_graph[node1][node2]['weight'] == pmg[node1][node2]['weight']
    #         assert process.process_model_graph[node1][node2]['time'] == pmg[node1][node2]['time']
    #         if process.gen_metrics:
    #             assert process.process_model_graph[node1][node2]['weight_normalized'] \
    #                    == pmg[node1][node2]['weight_normalized']
    #             assert process.process_model_graph[node1][node2]['time_normalized'] \
    #                    == pmg[node1][node2]['time_normalized']
    #
    #     pmg_by_cp = process.pmg_by_cp[0]
    #     assert pmg_by_cp[0] == 0
    #     assert pmg_by_cp[1] == ('activityA', 'activityB')
    #     assert pmg_by_cp[2] == 3
    #     assert pmg_by_cp[3] == 30.0
    #     assert pmg_by_cp[4] == 1.0
    #     assert pmg_by_cp[5] == 10.0

    # def test_set_case(self, process):
    #     assert not process.gp_creation
    #     print((datetime(2015, 5, 10, 22, 0, 0)-process.check_point).total_seconds())
    #     print(process.th)
    #
    #     process.set_case('1', 'activityA', datetime(2015, 5, 10, 22, 00, 00), 0)
    #     assert process.event_count == 1
    #     assert process.total_cases == {'1'}
    #     assert process.cp_cases == 1
    #     assert process.cases[0]
    #     case = process.cases[0]
    #     assert case.id == '1'
    #     assert np.isnan(case.time_distance)
    #     assert np.isnan(case.graph_distance)
    #     assert case.get_last_time() == datetime(2015, 5, 10, 22, 00, 00)
    #     assert case.get_trace() == ['activityA']
    #     assert case.get_timestamps() == [datetime(2015, 5, 10, 22, 00, 00)]
    #     assert process.gp_creation
    #
    #     process.set_case('1', 'activityB', datetime(2015, 5, 10, 22, 00, 10), 0)
    #     assert process.event_count == 2
    #     assert process.total_cases == {'1'}
    #     assert process.cp_cases == 1
    #     assert process.cases[0]
    #     case = process.cases[0]
    #     assert case.id == '1'
    #     assert np.isnan(case.time_distance)
    #     assert np.isnan(case.graph_distance)
    #     assert case.get_last_time() == datetime(2015, 5, 10, 22, 00, 10)
    #     assert case.get_trace() == ['activityA', 'activityB']
    #     assert case.get_timestamps() == [datetime(2015, 5, 10, 22, 00, 00),
    #                                      datetime(2015, 5, 10, 22, 00, 10)]
    #     assert process.gp_creation
    #
    #     process.set_case('2', 'activityA', datetime(2015, 5, 10, 22, 00, 00), 0)
    #     assert process.gp_creation
    #     process.set_case('2', 'activityB', datetime(2015, 5, 10, 22, 00, 10), 0)
    #     assert process.gp_creation
    #     process.set_case('2', 'activityC', datetime(2015, 5, 10, 22, 00, 20), 0)
    #     assert process.gp_creation