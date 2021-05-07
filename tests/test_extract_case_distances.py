from datetime import datetime
import networkx as nx
from cdesf2.data_structures import Case
from cdesf2.utils import initialize_graph, extract_case_distances
import pytest


class TestExtractCase:
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

    def test_extract_case_distances(self):
        case_list = []

        case = Case("1")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity D",
                "time:timestamp": datetime(2015, 5, 10, 8, 33, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity E",
                "time:timestamp": datetime(2015, 5, 10, 14, 6, 40),
            }
        )
        case_list.append(case)

        case = Case("2")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 1, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 14, 40, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity D",
                "time:timestamp": datetime(2015, 5, 10, 15, 5, 00),
            }
        )
        case_list.append(case)

        case = Case("3")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 13, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 13, 00),
            }
        )
        case_list.append(case)

        case = Case("4")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 20),
            }
        )
        case_list.append(case)

        case = Case("5")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("6")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 15, 30, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity F",
                "time:timestamp": datetime(2015, 5, 10, 15, 30, 00),
            }
        )
        case_list.append(case)

        case = Case("7")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity F",
                "time:timestamp": datetime(2015, 5, 11, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("8")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity F",
                "time:timestamp": datetime(2015, 5, 10, 11, 46, 40),
            }
        )
        case_list.append(case)

        case = Case("9")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity F",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("10")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity F",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        case = Case("11")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case_list.append(case)

        graph = initialize_graph(case_list)

        case = Case("12")
        case.add_event(
            {
                "concept:name": "Activity A",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity B",
                "time:timestamp": datetime(2015, 5, 10, 9, 18, 20),
            }
        )
        case.add_event(
            {
                "concept:name": "Activity C",
                "time:timestamp": datetime(2015, 5, 10, 10, 8, 20),
            }
        )

        distances = extract_case_distances(graph, case)
        assert distances.get("graph") == pytest.approx(0.05)
        assert distances.get("time") == pytest.approx(-1.42, rel=1e-2)

    def test_extract_case_distances_with_attributes(
        self,
        simple_graph_with_attributes: nx.DiGraph,
    ):
        case = Case("4")
        case.add_event(
            {
                "concept:name": "activityA",
                "time:timestamp": datetime(2015, 5, 10, 8, 00, 00),
                "attribute_one": 9,
                "attribute_two": 6,
            }
        )
        case.add_event(
            {
                "concept:name": "activityB",
                "time:timestamp": datetime(2015, 5, 10, 9, 18, 20),
                "attribute_one": 5,
                "attribute_two": 10,
            }
        )
        case.add_event(
            {
                "concept:name": "activityC",
                "time:timestamp": datetime(2015, 5, 10, 10, 8, 20),
                "attribute_one": 2,
                "attribute_two": 2,
            }
        )

        distances = extract_case_distances(
            simple_graph_with_attributes, case, ["attribute_one", "attribute_two"]
        )
        assert distances.get("attribute_one") == pytest.approx(0.4121, rel=1e-2)
        assert distances.get("attribute_two") == pytest.approx(0.2651, rel=1e-2)
