from datetime import datetime
import networkx as nx
import pytest
from cdesf2.data_structures import Case
from cdesf2.utils import initialize_graph, normalize_graph, merge_graphs


class TestGraph:
    @pytest.fixture
    def simple_graph(self):
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

        graph = nx.DiGraph(name='simple_graph')
        graph = initialize_graph(graph, case_list)

        return graph

    @pytest.fixture
    def complex_graph(self):
        case_list = []

        case = Case('1')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityD', datetime(2015, 5, 10, 8, 33, 20))
        case.set_activity('activityE', datetime(2015, 5, 10, 14, 6, 40))
        case_list.append(case)

        case = Case('2')
        case.set_activity('activityA', datetime(2015, 5, 10, 1, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 14, 40, 00))
        case.set_activity('activityD', datetime(2015, 5, 10, 15, 5, 00))
        case_list.append(case)

        case = Case('3')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 13, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 13, 00))
        case_list.append(case)

        case = Case('4')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 20))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 20))
        case_list.append(case)

        case = Case('5')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case_list.append(case)

        case = Case('6')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 15, 30, 00))
        case.set_activity('activityF', datetime(2015, 5, 10, 15, 30, 00))
        case_list.append(case)

        case = Case('7')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityF', datetime(2015, 5, 11, 8, 00, 00))
        case_list.append(case)

        case = Case('8')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityF', datetime(2015, 5, 10, 11, 46, 40))
        case_list.append(case)

        case = Case('9')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityF', datetime(2015, 5, 10, 8, 00, 00))
        case_list.append(case)

        case = Case('10')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityF', datetime(2015, 5, 10, 8, 00, 00))
        case_list.append(case)

        case = Case('11')
        case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityB', datetime(2015, 5, 10, 8, 00, 00))
        case.set_activity('activityC', datetime(2015, 5, 10, 8, 00, 00))
        case_list.append(case)

        graph = nx.DiGraph(name='complex_graph')
        graph = initialize_graph(graph, case_list)

        return graph

    def test_populate_graph(self, simple_graph, complex_graph):
        assert len(simple_graph.nodes()) == 4
        assert len(simple_graph.edges()) == 3
        assert simple_graph['activityA']['activityB']['weight'] == 3
        assert simple_graph['activityA']['activityB']['time'] == 30
        assert simple_graph['activityB']['activityC']['weight'] == 2
        assert simple_graph['activityB']['activityC']['time'] == 20
        assert simple_graph['activityC']['activityD']['weight'] == 1
        assert simple_graph['activityC']['activityD']['time'] == 10

        assert len(complex_graph.nodes()) == 6
        assert len(complex_graph.edges()) == 6
        assert complex_graph['activityA']['activityB']['weight'] == 10
        assert complex_graph['activityA']['activityB']['time'] == 50000
        assert complex_graph['activityB']['activityC']['weight'] == 9
        assert complex_graph['activityB']['activityC']['time'] == 27000
        assert complex_graph['activityC']['activityF']['weight'] == 5
        assert complex_graph['activityC']['activityF']['time'] == 100000
        assert complex_graph['activityA']['activityD']['weight'] == 1
        assert complex_graph['activityA']['activityD']['time'] == 2000
        assert complex_graph['activityB']['activityD']['weight'] == 1
        assert complex_graph['activityB']['activityD']['time'] == 1500
        assert complex_graph['activityD']['activityE']['weight'] == 1
        assert complex_graph['activityD']['activityE']['time'] == 20000

    def test_normalize_graph(self, simple_graph, complex_graph):
        simple_graph = normalize_graph(simple_graph)
        assert simple_graph['activityA']['activityB']['weight_normalized'] == 1
        assert simple_graph['activityA']['activityB']['time_normalized'] == 10
        assert simple_graph['activityB']['activityC']['weight_normalized'] == 2/3
        assert simple_graph['activityB']['activityC']['time_normalized'] == 10
        assert simple_graph['activityC']['activityD']['weight_normalized'] == 1/3
        assert simple_graph['activityC']['activityD']['time_normalized'] == 10

        complex_graph = normalize_graph(complex_graph)
        assert complex_graph['activityA']['activityB']['weight_normalized'] == 1
        assert complex_graph['activityA']['activityB']['time_normalized'] == 5000
        assert complex_graph['activityB']['activityC']['weight_normalized'] == 0.9
        assert complex_graph['activityB']['activityC']['time_normalized'] == 3000
        assert complex_graph['activityC']['activityF']['weight_normalized'] == 0.5
        assert complex_graph['activityC']['activityF']['time_normalized'] == 20000
        assert complex_graph['activityA']['activityD']['weight_normalized'] == 0.1
        assert complex_graph['activityA']['activityD']['time_normalized'] == 2000
        assert complex_graph['activityB']['activityD']['weight_normalized'] == 0.1
        assert complex_graph['activityB']['activityD']['time_normalized'] == 1500
        assert complex_graph['activityD']['activityE']['weight_normalized'] == 0.1
        assert complex_graph['activityD']['activityE']['time_normalized'] == 20000

    def test_merge_graphs(self, simple_graph, complex_graph):
        complex_graph = merge_graphs(complex_graph, simple_graph)
        assert complex_graph['activityA']['activityB']['weight'] == 12.5
        assert complex_graph['activityA']['activityB']['time'] == 50030
        assert complex_graph['activityB']['activityC']['weight'] == pytest.approx(10.55)
        assert complex_graph['activityB']['activityC']['time'] == 27020
        assert complex_graph['activityC']['activityF']['weight'] == 4.75
        assert complex_graph['activityC']['activityF']['time'] == 100000
        assert complex_graph['activityA']['activityD']['weight'] == 0.95
        assert complex_graph['activityA']['activityD']['time'] == 2000
        assert complex_graph['activityB']['activityD']['weight'] == 0.95
        assert complex_graph['activityB']['activityD']['time'] == 1500
        assert complex_graph['activityD']['activityE']['weight'] == 0.95
        assert complex_graph['activityD']['activityE']['time'] == 20000
        assert complex_graph['activityC']['activityD']['weight'] == 1
        assert complex_graph['activityC']['activityD']['time'] == 10

    # @pytest.fixture
    # def def_graph(self):
    #     def_graph = nx.Graph()
    #     tuple1 = ('node1', 'node2')
    #     tuple2 = ('node2', 'node3')
    #     tuple3 = ('node3', 'node4')
    #     def_graph.add_edge(*tuple1, weight=1, time=3, weight_norm=0, time_norm=0, count=1)
    #     def_graph.add_edge(*tuple2, weight=4, time=3, weight_norm=0, time_norm=0, count=2)
    #     def_graph.add_edge(*tuple3, weight=4, time=3, weight_norm=0, time_norm=0, count=2)
    #     return def_graph
    #
    # @pytest.fixture
    # def graph1(self):
    #     graph1 = nx.Graph()
    #     tuple1 = ('node1', 'node2')
    #     tuple2 = ('node2', 'node3')
    #     graph1.add_edge(*tuple1, weight=1, time=3, weight_norm=0, time_norm=0, count=1)
    #     graph1.add_edge(*tuple2, weight=4, time=3, weight_norm=0, time_norm=0, count=2)
    #     return graph1
    #
    # @pytest.fixture
    # def empty_graph(self):
    #     empty_graph = nx.Graph()
    #     return empty_graph
    #
    # def test_time_processing(self):
    #     time = graph.time_processing([['2020/06/30 13:00:00.0', '2020/06/30 14:00:00.0'],
    #                                   ['2020/06/30 14:00:00.0', '2020/06/30 16:00:00.0']])
    #     sub_list = time[0]
    #     date_obj = sub_list[0]
    #     assert type(time) == list
    #     assert type(sub_list) == list
    #     assert type(date_obj) == float
    #     assert time == [[3600.0], [7200.0]]
    #
    # def test_norm_graph(self, def_graph):
    #     normed_graph = graph.norm_graph(def_graph)
    #     assert normed_graph['node1']['node2']['weight_norm'] == 0.25
    #     assert normed_graph['node1']['node2']['time_norm'] == 3.0
    #     assert normed_graph['node2']['node3']['weight_norm'] == 1.0
    #     assert normed_graph['node2']['node3']['time_norm'] == 1.5
    #
    # def test_create_graph(self):
    #     time_list = [['2020/06/30 13:00:00.0', '2020/06/30 14:00:00.0'],
    #                  ['2020/06/30 14:00:00.0', '2020/06/30 16:00:00.0']]
    #     trace_list = ['activity1', 'activity2', 'activity3']
    #     new_graph = graph.create_graph(trace_list, time_list)
    #     assert new_graph['activity1']['activity2']
    #     assert new_graph['activity1']['activity2']['weight'] == 1
    #     assert new_graph['activity1']['activity2']['time'] == 3600.0
    #     assert new_graph['activity1']['activity2']['count'] == 1
    #     assert new_graph['activity2']['activity3']
    #     assert new_graph['activity2']['activity3']['weight'] == 1
    #     assert new_graph['activity2']['activity3']['time'] == 7200.0
    #     assert new_graph['activity3']['activity3']['count'] == 1
    #
    # def test_compute_feature(self, empty_graph, def_graph, graph1):
    #     trace1 = ['node1', 'node2', 'node3', 'node4']
    #     trace2 = ['node1', 'node2']
    #     trace3 = ['node4', 'node5']
    #     raw_time1 = ['2020/06/30 13:00:00.0', '2020/06/30 14:00:00.0',
    #                  '2020/06/30 16:00:00.0', '2020/06/30 16:30:00.0']
    #     raw_time2 = ['2020/06/30 13:00:00.0', '2020/06/30 14:00:00.0']
    #     raw_time3 = []
    #
    #     # if len(graph.edges) == 0
    #     gwd, twd = graph.compute_features(empty_graph, trace1, raw_time1)
    #     assert (gwd, twd) == (0, 0)
    #
    #     # twd: if g_sum == 0 && dif == 0
    #     # if (trace[i], trace[i+1]) NOT in graph.edges
    #     # gwd: if (len(trace)-1)<=1
    #     gwd, twd = graph.compute_features(graph1, trace3, raw_time3)
    #     assert gwd == 1
    #     assert twd == 0
    #
    #     # twd: if g_sum == 0 && dif != 0
    #     # if (trace[i], trace[i+1]) NOT in graph.edges
    #     # gwd: if (len(trace)-1)<=1
    #     gwd, twd = graph.compute_features(graph1, trace3, raw_time2)
    #     assert gwd == 1
    #     assert twd == 3.5563025007672873
    #
    #     # twd: if g_sum != 0 && dif != 0
    #     # if (trace[i], trace[i+1]) in graph.edges
    #     # gwd: if (len(trace)-1)<=1
    #     gwd, twd = graph.compute_features(graph1, trace2, raw_time2)
    #     assert gwd == 0.75
    #     assert twd == 3.0788191830988487
    #
    #     # twd: if g_sum != 0 && dif == 0
    #     # if (trace[i], trace[i+1]) in graph.edges
    #     # gwd: if (len(trace)-1)>1
    #     gwd, twd = graph.compute_features(def_graph, trace1, raw_time3)
    #     assert gwd == 0.25
    #     assert twd == 0
    #
    #     # twd: if g_sum != 0 && dif != 0
    #     # if (trace[i], trace[i+1]) in graph.edges
    #     # gwd: if (len(trace)-1)>1
    #     gwd, twd = graph.compute_features(def_graph, trace1, raw_time1)
    #     assert gwd == 0.25
    #     assert twd == 3.3220124385824006
    #
    #     # twd: if g_sum != 0 && dif == 0
    #     # if (trace[i], trace[i+1]) in graph.edges
    #     # gwd: if (len(trace)-1)<=1
    #     gwd, twd = graph.compute_features(graph1, trace2, raw_time3)
    #     assert gwd == 0.75
    #     assert twd == 0
    #
    # def test_merge_graphs(self, def_graph, graph1):
    #     graph1.add_edge('node5', 'node6', weight=2, time=3, weight_norm=0, time_norm=0, count=2)
    #     graph.merge_graphs(def_graph, graph1)
    #
    #     # weight
    #     assert def_graph['node1']['node2']['weight'] == 1.95
    #     assert def_graph['node2']['node3']['weight'] == 7.8
    #     assert def_graph['node3']['node4']['weight'] == 3.8  # because the (n3,n4) does not exist in graph1
    #     assert def_graph['node5']['node6']['weight'] == 2
    #
    #     # time
    #     assert def_graph['node1']['node2']['time'] == 6
    #     assert def_graph['node3']['node4']['time'] == 3
    #     assert def_graph['node5']['node6']['time'] == 3
    #
    #     # count
    #     assert def_graph['node1']['node2']['count'] == 2  # first node count = 1. 1+1=2
    #     assert def_graph['node3']['node4']['count'] == 2
    #     assert def_graph['node5']['node6']['count'] == 2
