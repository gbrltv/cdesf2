from datetime import datetime
from os import path

import networkx as nx
import pytest
from cdesf.clustering import DenStream
from cdesf.core import CDESF
from cdesf.data_structures import Case
from cdesf.utils import (
    calculate_case_distances,
    initialize_graph,
    normalize_graph,
    read_csv,
    read_xes,
)


class TestCdesf:
    @pytest.fixture
    def process(self):
        process = CDESF(
            name="test",
            time_horizon=43200,
            lambda_=0.15,
            beta=0.3,
            epsilon=0.1,
            mu=4,
            stream_speed=1000,
            gen_plot=False,
            gen_metrics=False,
        )
        return process

    @pytest.fixture
    def cases_list(self):
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

        return [case_3, case_2, case_1]

    def test_initial_value(self, process):
        assert isinstance(process.name, str)
        assert process.name == "test"
        assert isinstance(process.gen_plot, bool)
        assert not process.gen_plot
        assert isinstance(process.gen_metrics, bool)
        assert not process.gen_metrics
        assert process.event_index == 0
        assert process.total_cases == set()
        assert process.cases == []
        assert isinstance(process.time_horizon, int)
        assert process.time_horizon == 43200
        assert not process.initialized
        assert process.cp_count == 0
        assert process.nyquist == 0
        assert process.check_point_cases == 0

        assert isinstance(process.process_model_graph, nx.DiGraph)
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

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

    def test_initialize_case_metrics(self, process: CDESF, cases_list: "list[Case]"):
        process.initialize_case_metrics()
        assert process.cases == []

        process.cases = cases_list

        case_3 = process.cases[0]
        assert case_3.distances.get("graph") is None
        assert case_3.distances.get("graph") is None
        case_2 = process.cases[1]
        assert case_2.distances.get("graph") is None
        assert case_2.distances.get("graph") is None
        case_1 = process.cases[2]
        assert case_1.distances.get("graph") is None
        assert case_1.distances.get("graph") is None

        process.initialize_case_metrics()

        case_3 = process.cases[0]
        assert case_3.distances["graph"] == 0
        assert case_3.distances["time"] == 0
        case_2 = process.cases[1]
        assert case_2.distances["graph"] == 0
        assert case_2.distances["time"] == 0
        case_1 = process.cases[2]
        assert case_1.distances["graph"] == 0
        assert case_1.distances["time"] == 0

        process.cases = cases_list
        process.process_model_graph = initialize_graph(process.cases)
        process.initialize_case_metrics()
        case_3 = process.cases[0]

        # Verify that the distances are applied.
        distances_for_case_3 = calculate_case_distances(
            process.process_model_graph, case_3
        )
        assert case_3.distances["graph"] == distances_for_case_3["graph"]
        assert case_3.distances["time"] == distances_for_case_3["time"]

        distances_for_case_2 = calculate_case_distances(
            process.process_model_graph, case_2
        )
        case_2 = process.cases[1]
        assert case_2.distances["graph"] == distances_for_case_2["graph"]
        assert case_2.distances["time"] == distances_for_case_2["time"]

        distances_for_case_1 = calculate_case_distances(
            process.process_model_graph, case_1
        )
        case_1 = process.cases[2]
        assert case_1.distances["graph"] == distances_for_case_1["graph"]
        assert case_1.distances["time"] == distances_for_case_1["time"]

    def test_get_case(self, process):
        case1 = Case("1")
        case2 = Case("2")
        case3 = Case("3")
        case4 = Case("4")
        process.cases = [case1, case2, case3, case4]

        case = process.get_case_index("1")
        assert case == 0
        case = process.get_case_index("3")
        assert case == 2

        process.cases = [case4, case1, case3, case2]
        case = process.get_case_index("1")
        assert case == 1
        case = process.get_case_index("2")
        assert case == 3

        case = process.get_case_index("5")
        assert case is None

    def test_release_cases_from_memory(self, process):
        case1 = Case("1")
        case2 = Case("2")
        case3 = Case("3")
        case4 = Case("4")
        case4.add_event(
            {
                "concept:name": "activity1",
                "time:timestamp": datetime(2015, 5, 10, 8, 30, 00),
            }
        )
        case1.add_event(
            {
                "concept:name": "activity1",
                "time:timestamp": datetime(2015, 5, 10, 9, 00, 00),
            }
        )
        case3.add_event(
            {
                "concept:name": "activity1",
                "time:timestamp": datetime(2015, 5, 10, 9, 30, 00),
            }
        )
        case2.add_event(
            {
                "concept:name": "activity1",
                "time:timestamp": datetime(2015, 5, 10, 10, 00, 00),
            }
        )

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
            assert initialize_graph([])

        case4 = Case("4")
        case4.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case4.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            }
        )
        case4.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )

        process.cases = cases_list
        process.cases.insert(0, case4)
        process.process_model_graph = initialize_graph(process.cases)
        pmg = initialize_graph(cases_list)
        for case in cases_list:
            case.distances = calculate_case_distances(pmg, case)
        case_4 = process.cases[0]

        process.initialize_cdesf()

        distances_for_case_4 = calculate_case_distances(pmg, case_4)
        assert case_4.distances["graph"] == distances_for_case_4["graph"]
        assert case_4.distances["time"] == distances_for_case_4["time"]

        assert len(process.denstream.p_micro_clusters) == 1
        assert process.denstream.all_cases.keys() == {"1", "4"}
        new_p_mc = process.denstream.p_micro_clusters[0]
        assert new_p_mc.weight == 2
        assert new_p_mc.creation_time == 0
        assert new_p_mc.lambda_ == 0.15

        assert process.initialized

    def test_set_case(self, process: CDESF):
        assert process.check_point_cases == 0
        assert process.cases == []
        process.set_case(
            "5",
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        assert process.check_point_cases == 1
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == "5"
        assert case_test.events[0].get("concept:name") == "activityA"
        assert case_test.events[0].get("time:timestamp") == datetime(
            2015, 5, 10, 8, 00, 00
        )

        case1 = Case("1")
        case2 = Case("2")
        case3 = Case("3")
        case4 = Case("4")
        process.check_point_cases = 0
        process.cases = [case1, case2, case3, case4]
        process.set_case(
            "3",
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        assert process.check_point_cases == 0
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == "3"
        assert case_test.events[0].get("concept:name") == "activityA"
        assert case_test.events[0].get("time:timestamp") == datetime(
            2015, 5, 10, 8, 00, 00
        )

    def test_check_point_update(self, process, cases_list):
        assert process.cp_count == 0
        process.check_point_update()

        assert process.cp_count == 1
        assert process.check_point_cases == 0
        assert process.nyquist == 0
        assert len(process.process_model_graph.edges) == 0
        assert len(process.process_model_graph.nodes) == 0

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
        assert process.nyquist == 1
        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        check_point_graph = initialize_graph(process.cases)
        for node1, node2, _ in process.process_model_graph.edges.data():
            assert (
                process.process_model_graph[node1][node2]
                == check_point_graph[node1][node2]
            )
            assert (
                process.process_model_graph[node1][node2]["weight"]
                == check_point_graph[node1][node2]["weight"]
            )
            assert (
                process.process_model_graph[node1][node2]["time"]
                == check_point_graph[node1][node2]["time"]
            )

        # self.cases > nyquist
        # nyquist = 1
        # process_model_graph = 1
        process.process_model_graph = initialize_graph([cases_list[2]])
        pmg = process.process_model_graph
        assert pmg["activityA"]["activityB"]
        assert pmg["activityB"]["activityC"]
        with pytest.raises(Exception):
            assert pmg["activityC"]["activityD"]
            assert pmg["activityD"]["activityE"]
        case3 = cases_list[0]
        process.cases = cases_list
        process.check_point_cases = 1
        process.nyquist = 1

        process.check_point_update()

        assert process.cp_count == 5
        assert process.cases == [case3]
        assert process.check_point_cases == 0
        assert process.nyquist == 1
        assert len(process.process_model_graph.edges) == 3
        assert len(process.process_model_graph.nodes) == 4
        assert pmg["activityA"]["activityB"]
        assert pmg["activityB"]["activityC"]
        assert pmg["activityC"]["activityD"]
        check_point_graph = initialize_graph(process.cases)
        for node1, node2, _ in process.process_model_graph.edges.data():
            pmg[node1][node2]["weight"] *= 0.95
            pmg[node1][node2]["weight"] += check_point_graph[node1][node2]["weight"]
            pmg[node1][node2]["time"] += check_point_graph[node1][node2]["time"]
            pmg = normalize_graph(pmg)
            assert (
                process.process_model_graph[node1][node2]["weight"]
                == pmg[node1][node2]["weight"]
            )
            assert (
                process.process_model_graph[node1][node2]["weight_normalized"]
                == pmg[node1][node2]["weight_normalized"]
            )
            assert (
                process.process_model_graph[node1][node2]["time"]
                == pmg[node1][node2]["time"]
            )
            assert (
                process.process_model_graph[node1][node2]["time_normalized"]
                == pmg[node1][node2]["time_normalized"]
            )

        # self.cases > nyquist
        # nyquist = 2
        # process_model_graph = 1
        case2 = cases_list[1]
        case3 = cases_list[0]

        process.process_model_graph = initialize_graph([cases_list[1]])
        pmg = process.process_model_graph
        assert pmg["activityA"]["activityB"]
        with pytest.raises(Exception):
            assert pmg["activityC"]["activityD"]
            assert pmg["activityD"]["activityE"]

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
        assert process.process_model_graph["activityA"]["activityB"]
        assert process.process_model_graph["activityB"]["activityC"]
        assert process.process_model_graph["activityC"]["activityD"]
        check_point_graph = initialize_graph(process.cases)
        for node1, node2, _ in process.process_model_graph.edges.data():
            pmg[node1][node2]["weight"] *= 0.95
            pmg[node1][node2]["weight"] += check_point_graph[node1][node2]["weight"]
            pmg[node1][node2]["time"] += check_point_graph[node1][node2]["time"]
            pmg = normalize_graph(pmg)
            assert (
                process.process_model_graph[node1][node2]["weight"]
                == pmg[node1][node2]["weight"]
            )
            assert (
                process.process_model_graph[node1][node2]["weight_normalized"]
                == pmg[node1][node2]["weight_normalized"]
            )
            assert (
                process.process_model_graph[node1][node2]["time"]
                == pmg[node1][node2]["time"]
            )
            assert (
                process.process_model_graph[node1][node2]["time_normalized"]
                == pmg[node1][node2]["time_normalized"]
            )

        # self.cases > nyquist
        # nyquist = 2
        # process_model_graph = 2
        case2 = cases_list[1]
        case3 = cases_list[0]

        process.process_model_graph = initialize_graph([cases_list[1], cases_list[2]])
        pmg = process.process_model_graph
        assert pmg["activityA"]["activityB"]
        assert pmg["activityB"]["activityC"]
        with pytest.raises(Exception):
            assert pmg["activityC"]["activityD"]
            assert pmg["activityD"]["activityE"]

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
        assert process.process_model_graph["activityA"]["activityB"]
        assert process.process_model_graph["activityB"]["activityC"]
        assert process.process_model_graph["activityC"]["activityD"]
        check_point_graph = initialize_graph(process.cases)
        for node1, node2, _ in process.process_model_graph.edges.data():
            pmg[node1][node2]["weight"] *= 0.95
            pmg[node1][node2]["weight"] += check_point_graph[node1][node2]["weight"]
            pmg[node1][node2]["time"] += check_point_graph[node1][node2]["time"]
            pmg = normalize_graph(pmg)
            assert (
                process.process_model_graph[node1][node2]["weight"]
                == pmg[node1][node2]["weight"]
            )
            assert (
                process.process_model_graph[node1][node2]["weight_normalized"]
                == pmg[node1][node2]["weight_normalized"]
            )
            assert (
                process.process_model_graph[node1][node2]["time"]
                == pmg[node1][node2]["time"]
            )
            assert (
                process.process_model_graph[node1][node2]["time_normalized"]
                == pmg[node1][node2]["time_normalized"]
            )

    def test_process_event(self, process: CDESF):
        process.gen_metrics = True
        assert process.event_index == 0
        assert process.cases == []
        assert process.check_point == datetime(2010, 1, 1)
        process.check_point = datetime(2015, 5, 10, 8, 22, 53)

        process.process_event(
            {
                "case:concept:name": "1",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        assert process.event_index == 0
        assert process.total_cases == {"1"}

        assert process.check_point_cases == 1
        assert process.cases
        case_test = process.cases[0]
        assert case_test.id == "1"
        event = case_test.events[0]
        assert event.get("concept:name") == "activityA"
        assert event.get("time:timestamp") == datetime(2015, 5, 10, 8, 00, 00)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert not process.initialized

        process.process_event(
            {
                "case:concept:name": "2",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        process.process_event(
            {
                "case:concept:name": "1",
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            },
        )
        process.process_event(
            {
                "case:concept:name": "3",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        process.process_event(
            {
                "case:concept:name": "3",
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
            },
        )
        process.process_event(
            {
                "case:concept:name": "4",
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            },
        )
        process.process_event(
            {
                "case:concept:name": "3",
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            },
        )
        process.process_event(
            {
                "case:concept:name": "4",
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            },
        )
        process.process_event(
            {
                "case:concept:name": "5",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            },
        )
        process.process_event(
            {
                "case:concept:name": "5",
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
            },
        )

        assert process.check_point_cases == 5
        assert process.total_cases == {"1", "2", "3", "4", "5"}
        assert process.cases
        assert len(process.cases) == 5
        case_test = process.cases[0]
        assert case_test.id == "5"
        event = case_test.events[0]
        assert event.get("concept:name") == "activityA"
        assert event.get("time:timestamp") == datetime(2015, 5, 10, 8, 00, 00)
        event = case_test.events[1]
        assert event.get("concept:name") == "activityD"
        assert event.get("time:timestamp") == datetime(2015, 5, 10, 8, 00, 30)
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert not process.initialized

        process.process_event(
            {
                "case:concept:name": "4",
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 40),
            },
        )
        assert process.check_point == datetime(2015, 5, 10, 8, 22, 53)
        assert process.check_point_cases == 5
        assert not process.initialized

        process.process_event(
            {
                "case:concept:name": "4",
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 21, 00, 40),
            },
        )
        assert process.initialized
        assert process.check_point_cases == 5
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        # print(list(process.process_model_graph.edges(data=True)))
        assert len(process.process_model_graph.edges) == 6
        # assert len(process.denstream.p_micro_clusters) == 2
        assert len(process.denstream.o_micro_clusters) == 0
        assert process.nyquist == 10

        process.process_event(
            {
                "case:concept:name": "1",
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 21, 00, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "1",
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 21, 50, 50),
            },
        )
        assert process.check_point_cases == 5
        assert process.total_cases == {"1", "2", "3", "4", "5"}
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5"}
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        # assert len(process.denstream.p_micro_clusters) == 2
        # assert len(process.denstream.o_micro_clusters) == 2
        assert process.nyquist == 10

        process.process_event(
            {
                "case:concept:name": "1",
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 21, 50, 50),
            },
        )
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert process.check_point_cases == 5
        assert process.total_cases == {"1", "2", "3", "4", "5"}
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5"}
        assert process.check_point == datetime(2015, 5, 10, 21, 00, 40)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        # assert len(process.denstream.p_micro_clusters) == 3
        # assert len(process.denstream.o_micro_clusters) == 1
        assert process.nyquist == 10

        process.process_event(
            {
                "case:concept:name": "6",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 11, 10, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "7",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 11, 10, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "8",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 11, 10, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "9",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 11, 10, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "10",
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 11, 10, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "2",
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 12, 22, 50, 50),
            },
        )
        process.process_event(
            {
                "case:concept:name": "2",
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 13, 22, 50, 50),
            },
        )

        assert process.cp_count == 1
        assert path.isfile(
            f"output/metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json"
        )

        assert process.check_point_cases == 0
        assert process.check_point == datetime(2015, 5, 13, 22, 50, 50)
        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 6
        # assert len(process.denstream.p_micro_clusters) == 4
        # assert len(process.denstream.o_micro_clusters) == 1
        assert process.total_cases == {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
        }
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5"}
        assert process.nyquist == 10
        assert len(process.cases) == 10

        process.process_event(
            {
                "case:concept:name": "11",
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 13, 23, 00, 00),
            },
        )
        assert len(process.cases) == 11
        assert process.total_cases == {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
        }
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5"}
        assert process.nyquist == 10
        process.process_event(
            {
                "case:concept:name": "11",
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 14, 12, 00, 00),
            },
        )

        assert process.cp_count == 2
        assert path.isfile(
            f"output/metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json"
        )

        assert len(process.cases) == 10
        assert process.total_cases == {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
        }
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5", "11"}
        assert process.nyquist == 10

        process.process_event(
            {
                "case:concept:name": "11",
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 15, 1, 00, 00),
            },
        )
        assert process.cp_count == 3
        assert path.isfile(
            f"output/metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json"
        )

        assert len(process.cases) == 10
        assert process.cases[0].id == "11"
        assert process.cases[1].id == "2"
        assert process.total_cases == {
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
        }
        assert process.denstream.all_cases.keys() == {"1", "2", "3", "4", "5", "11"}
        assert process.nyquist == 10

        assert len(process.process_model_graph.nodes) == 5
        assert len(process.process_model_graph.edges) == 7
        # assert len(process.denstream.p_micro_clusters) == 4
        # assert len(process.denstream.o_micro_clusters) == 3

    # def test_save_pmg_on_check_point(self, process, cases_list):
    #     assert process.metrics.path_to_pmg_metrics == f'metrics/{process.name}_process_model_graphs'
    #
    #     assert len(process.process_model_graph.edges) == 0
    #     process.metrics.save_pmg_on_check_point()
    #     assert process.cp_count == 0
    #     assert path.exists('./metrics')
    #     assert not path.isfile(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')
    #
    #     process.cases = cases_list
    #     process.process_model_graph = initialize_graph(process.cases)
    #     pmg = initialize_graph(process.cases)
    #     pmg = normalize_graph(pmg)
    #     process.initialize_case_metrics()
    #     process.save_pmg_on_check_point()
    #
    #     assert process.cp_count == 0
    #     assert path.exists('./metrics')
    #     assert path.isfile(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')
    #     invalid_cp_count = process.cp_count + 1
    #     assert not path.isfile(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{invalid_cp_count}.json')
    #     with open(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json') as file:
    #         data = json.load(file)
    #
    #     nodes = data['nodes']
    #     edges = data['links']
    #     activities_ids = [node['id'] for node in nodes]
    #     assert len(edges) == len(pmg.edges)
    #     assert activities_ids == list(pmg)
    #     i = 0
    #     for node1, node2, data in pmg.edges(data=True):
    #         edge = edges[i]
    #         edge_nodes = (edge['source'], edge['target'])
    #         assert edge_nodes == (node1, node2)
    #         assert edge['weight'] == data['weight']
    #         assert edge['time'] == data['time']
    #         assert edge['weight_normalized'] == data['weight_normalized']
    #         assert edge['time_normalized'] == data['time_normalized']
    #         i += 1
    #
    #     remove(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')
    #
    #     process.cp_count = 1
    #     assert not path.isfile(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')
    #
    #     pmg = normalize_graph(pmg)
    #     process.save_pmg_on_check_point()
    #
    #     assert path.isfile(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')
    #     with open(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json') as file:
    #         data = json.load(file)
    #
    #     nodes = data['nodes']
    #     edges = data['links']
    #     activities_ids = [d['id'] for d in nodes]
    #     assert len(edges) == len(pmg.edges)
    #     assert activities_ids == list(pmg)
    #     i = 0
    #     for node1, node2, data in pmg.edges(data=True):
    #         edge = edges[i]
    #         edge_nodes = (edge['source'], edge['target'])
    #         assert edge_nodes == (node1, node2)
    #         assert edge['weight'] == data['weight']
    #         assert edge['time'] == data['time']
    #         assert edge['weight_normalized'] == data['weight_normalized']
    #         assert edge['time_normalized'] == data['time_normalized']
    #         i += 1
    #
    #     remove(f'./metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json')

    def test_run(self, process):
        stream = read_csv("demo/Detail_Supplier_IW-Frozen.csv")
        process.run(stream)

        assert process.initialized
        assert path.isfile(
            f"output/metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json"
        )

    def test_run_xes(self, process):
        stream = read_xes("demo/running-example.xes")
        process.run(stream)

        assert process.initialized
        assert path.isfile(
            f"output/metrics/{process.name}_process_model_graphs/process_model_graph_{process.cp_count}.json"
        )
