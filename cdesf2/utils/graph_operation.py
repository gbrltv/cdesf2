import networkx as nx
from typing import List
from ..data_structures.case import Case
from ..utils import time_difference, extract_cases_time_and_trace


def normalize_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Time and weight normalization for each edge in the graph.
    Time normalization is the mean time of an edge.
    Trace normalization is based on the graph weights

    Parameters
    --------------------------------------
    graph: nx.DiGraph,
        Graph to be normalized
    Returns
    --------------------------------------
    graph: nx.DiGraph,
        Normalized graph
    """
    max_weight = max([attributes['weight'] for n1, n2, attributes in graph.edges(data=True)])
    for node1, node2, data in graph.edges(data=True):
        data['weight_normalized'] = data['weight'] / max_weight
        data['time_normalized'] = data['time'] / data['weight']

    return graph


def initialize_graph(graph: nx.DiGraph, case_list: List[Case]) -> nx.DiGraph:
    """
    Initialize a graph based on the weights and time differences from a list of cases

    Parameters
    --------------------------------------
    graph: nx.DiGraph,
        Graph to be initialized
    case_list: List[Case],
        List of cases used to initialize the graph
    Returns
    --------------------------------------
    graph: nx.DiGraph,
        Initialized graph
    """
    trace_list, time_list = extract_cases_time_and_trace(case_list)

    time_list = time_difference(time_list)

    for trace, time in zip(trace_list, time_list):
        for i in range(len(trace)-1):
            edges = (trace[i], trace[i+1])
            if edges not in graph.edges:
                graph.add_edge(*edges, weight=1, time=time[i])
            else:
                graph[edges[0]][edges[1]]['weight'] += 1
                graph[edges[0]][edges[1]]['time'] += time[i]

    return normalize_graph(graph)


def merge_graphs(process_model_graph: nx.DiGraph, check_point_graph: nx.DiGraph) -> nx.DiGraph:
    """
    Receives two graphs and merge them.
    The first is the PMG (process model graph) and the second it the CP (check point) graph.
    The PMG, then, incorporates the second graph.
    Before the merge, 5% of the PMG's weight is decayed.

    Parameters
    --------------------------------------
    process_model_graph: nx.DiGraph,
        PMG graph
    check_point_graph: nx.DiGraph,
        CP graph
    Returns
    --------------------------------------
    process_model_graph: nx.DiGraph,
        PMG after merge
    """
    for node1, node2, data in process_model_graph.edges(data=True):
        data['weight'] *= 0.95

    for node1, node2, data in check_point_graph.edges(data=True):
        path = (node1, node2)
        if path in process_model_graph.edges:
            process_model_graph[node1][node2]['weight'] += data['weight']
            process_model_graph[node1][node2]['time'] += data['time']
        else:
            process_model_graph.add_edge(*path, weight=data['weight'], time=data['time'])

    return normalize_graph(process_model_graph)
