from datetime import datetime
import networkx as nx
import pytest
from cdesf.data_structures import Case
from cdesf.utils import initialize_graph, normalize_graph, merge_graphs


class TestGraph:
    @pytest.fixture
    def simple_graph(self):
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

        graph = initialize_graph(case_list)

        return graph

    @pytest.fixture
    def simple_graph_with_attributes(self):
        case_list = []

        case = Case("1")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
                "attribute_one": 10,
                "attribute_two": 5,
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
                "attribute_one": 0,
                "attribute_two": 7,
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
                "attribute_one": 2,
                "attribute_two": 4,
            }
        )
        case_list.append(case)

        case = Case("2")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
                "attribute_one": 3,
                "attribute_two": 10,
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
                "attribute_one": 6,
                "attribute_two": 2,
            }
        )
        case_list.append(case)

        case = Case("3")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
                "attribute_one": 8,
                "attribute_two": 4,
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 10),
                "attribute_one": 1,
                "attribute_two": 12,
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
                "attribute_one": 2,
                "attribute_two": 2,
            }
        )
        case.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 30),
                "attribute_one": 1,
                "attribute_two": 1,
            }
        )
        case_list.append(case)

        graph = initialize_graph(case_list, ["attribute_one", "attribute_two"])

        return graph

    @pytest.fixture
    def complex_graph(self):
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
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 8, 33, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "activityE",
                "time:timestamp": datetime(2015, 5, 10, 14, 6, 40),
            }
        )
        case_list.append(case)

        case = Case("2")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 1, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 14, 40, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityD",
                "time:timestamp": datetime(2015, 5, 10, 15, 5, 00),
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
                "time:timestamp": datetime(2015, 5, 10, 8, 13, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 13, 00),
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
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_list.append(case)

        case = Case("5")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("6")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 15, 30, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 10, 15, 30, 00),
            }
        )
        case_list.append(case)

        case = Case("7")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 11, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("8")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 10, 11, 46, 40),
            }
        )
        case_list.append(case)

        case = Case("9")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("10")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityF",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("11")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        graph = initialize_graph(case_list)

        return graph

    def test_populate_graph(
        self, simple_graph, simple_graph_with_attributes: nx.DiGraph, complex_graph
    ):
        assert len(simple_graph.nodes()) == 4
        assert len(simple_graph.edges()) == 3
        assert simple_graph["activityA"]["activityB"]["weight"] == 3
        assert simple_graph["activityA"]["activityB"]["time"] == 30
        assert simple_graph["activityB"]["activityC"]["weight"] == 2
        assert simple_graph["activityB"]["activityC"]["time"] == 20
        assert simple_graph["activityC"]["activityD"]["weight"] == 1
        assert simple_graph["activityC"]["activityD"]["time"] == 10

        assert len(simple_graph_with_attributes.nodes()) == 4
        assert len(simple_graph_with_attributes.edges()) == 3
        assert simple_graph_with_attributes["activityA"]["activityB"]["weight"] == 3
        assert simple_graph_with_attributes["activityA"]["activityB"]["time"] == 30
        # TODO: Maybe (just maybe) the node should hold an array with all different values, then the distances function can calculate the distance with the new one?
        assert simple_graph_with_attributes.nodes["activityA"]["attribute_one"] == [
            10,
            3,
            8,
        ]
        assert simple_graph_with_attributes.nodes["activityA"]["attribute_two"] == [
            5,
            10,
            4,
        ]
        assert simple_graph_with_attributes.nodes["activityB"]["attribute_one"] == [
            0,
            6,
            1,
        ]

        assert len(complex_graph.nodes()) == 6
        assert len(complex_graph.edges()) == 6
        assert complex_graph["activityA"]["activityB"]["weight"] == 10
        assert complex_graph["activityA"]["activityB"]["time"] == 50000
        assert complex_graph["activityB"]["activityC"]["weight"] == 9
        assert complex_graph["activityB"]["activityC"]["time"] == 27000
        assert complex_graph["activityC"]["activityF"]["weight"] == 5
        assert complex_graph["activityC"]["activityF"]["time"] == 100000
        assert complex_graph["activityA"]["activityD"]["weight"] == 1
        assert complex_graph["activityA"]["activityD"]["time"] == 2000
        assert complex_graph["activityB"]["activityD"]["weight"] == 1
        assert complex_graph["activityB"]["activityD"]["time"] == 1500
        assert complex_graph["activityD"]["activityE"]["weight"] == 1
        assert complex_graph["activityD"]["activityE"]["time"] == 20000

    def test_normalize_graph(self, simple_graph, complex_graph):
        simple_graph = normalize_graph(simple_graph)
        assert simple_graph["activityA"]["activityB"]["weight_normalized"] == 1
        assert simple_graph["activityA"]["activityB"]["time_normalized"] == 10
        assert simple_graph["activityB"]["activityC"]["weight_normalized"] == 2 / 3
        assert simple_graph["activityB"]["activityC"]["time_normalized"] == 10
        assert simple_graph["activityC"]["activityD"]["weight_normalized"] == 1 / 3
        assert simple_graph["activityC"]["activityD"]["time_normalized"] == 10

        complex_graph = normalize_graph(complex_graph)
        assert complex_graph["activityA"]["activityB"]["weight_normalized"] == 1
        assert complex_graph["activityA"]["activityB"]["time_normalized"] == 5000
        assert complex_graph["activityB"]["activityC"]["weight_normalized"] == 0.9
        assert complex_graph["activityB"]["activityC"]["time_normalized"] == 3000
        assert complex_graph["activityC"]["activityF"]["weight_normalized"] == 0.5
        assert complex_graph["activityC"]["activityF"]["time_normalized"] == 20000
        assert complex_graph["activityA"]["activityD"]["weight_normalized"] == 0.1
        assert complex_graph["activityA"]["activityD"]["time_normalized"] == 2000
        assert complex_graph["activityB"]["activityD"]["weight_normalized"] == 0.1
        assert complex_graph["activityB"]["activityD"]["time_normalized"] == 1500
        assert complex_graph["activityD"]["activityE"]["weight_normalized"] == 0.1
        assert complex_graph["activityD"]["activityE"]["time_normalized"] == 20000

    def test_merge_graphs(self, simple_graph, complex_graph):
        complex_graph = merge_graphs(complex_graph, simple_graph)
        assert complex_graph["activityA"]["activityB"]["weight"] == 12.5
        assert complex_graph["activityA"]["activityB"]["time"] == 50030
        assert complex_graph["activityB"]["activityC"]["weight"] == pytest.approx(10.55)
        assert complex_graph["activityB"]["activityC"]["time"] == 27020
        assert complex_graph["activityC"]["activityF"]["weight"] == 4.75
        assert complex_graph["activityC"]["activityF"]["time"] == 100000
        assert complex_graph["activityA"]["activityD"]["weight"] == 0.95
        assert complex_graph["activityA"]["activityD"]["time"] == 2000
        assert complex_graph["activityB"]["activityD"]["weight"] == 0.95
        assert complex_graph["activityB"]["activityD"]["time"] == 1500
        assert complex_graph["activityD"]["activityE"]["weight"] == 0.95
        assert complex_graph["activityD"]["activityE"]["time"] == 20000
        assert complex_graph["activityC"]["activityD"]["weight"] == 1
        assert complex_graph["activityC"]["activityD"]["time"] == 10

    def test_merge_graphs_with_attributes(self, simple_graph_with_attributes):
        new_graph = merge_graphs(
            simple_graph_with_attributes, simple_graph_with_attributes
        )

        assert new_graph.nodes["activityA"]["attribute_one"] == [10, 3, 8, 10, 3, 8]
        assert new_graph.nodes["activityA"]["attribute_two"] == [5, 10, 4, 5, 10, 4]
        assert new_graph.nodes["activityB"]["attribute_one"] == [0, 6, 1, 0, 6, 1]
