from datetime import datetime
import networkx as nx
from cdesf2.data_structures import Case
from cdesf2.utils import initialize_graph, normalize_graph
from cdesf2.utils import extract_case_distances
import pytest


def test_extract_case_distances():
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

    graph = nx.DiGraph()
    graph = initialize_graph(graph, case_list)

    case = Case('12')
    case.set_activity('activityA', datetime(2015, 5, 10, 8, 00, 00))
    case.set_activity('activityB', datetime(2015, 5, 10, 9, 18, 20))
    case.set_activity('activityC', datetime(2015, 5, 10, 10, 8, 20))

    trace_distance, time_distance = extract_case_distances(graph, case)
    assert trace_distance == pytest.approx(0.05)
    assert time_distance == pytest.approx(-1.42, rel=1e-2)
