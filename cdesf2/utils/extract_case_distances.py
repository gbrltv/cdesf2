from math import log10
import networkx as nx
from typing import Tuple
from ..data_structures import Case
from .time_difference import time_difference


def extract_case_distances(graph: nx.DiGraph, case: Case) -> "dict[str, float]":
    trace = case.get_trace()
    timestamps = case.get_timestamps()
    timestamp_differences = time_difference([timestamps])[0]

    if not graph.edges or len(trace) < 2:
        return {"graph": 0, "time": 0}

    graph_time = []
    trace_weight = 0
    for index in range(len(trace) - 1):
        if (trace[index], trace[index + 1]) in graph.edges:
            graph_time.append(graph[trace[index]][trace[index + 1]]["time_normalized"])
            trace_weight += (
                1 - graph[trace[index]][trace[index + 1]]["weight_normalized"]
            )
        else:
            graph_time.append(0)
            trace_weight += 1

    lent = len(trace) - 1
    if lent > 1:
        # trace has more than one edge (transition), graph distance is the division of the sum by the number of edges
        graph_distance = trace_weight / lent
    else:
        # trace has only one edge, graph distance is the sum
        graph_distance = trace_weight

    difference = 0
    time = time_difference([case.get_timestamps()])[0]
    for i in range(len(graph_time)):
        difference += abs(graph_time[i] - time[i])
    if difference == 0:
        # trace and graph have the exact same time distribution
        return {"graph": graph_distance, "time": 0}

    graph_time_sum = sum(graph_time)
    if graph_time_sum == 0:
        # graph time sum is 0, returns the log of the difference
        return {"graph": graph_distance, "time": log10(difference)}
    else:
        # graph time sum and difference are not zero, original equation performed (most likely)
        return {"graph": graph_distance, "time": log10(difference / graph_time_sum)}


# def OLD_extract_case_distances(graph: nx.DiGraph, case: Case) -> Tuple[float, float]:
#     """
#     Receives a graph and a case and computes the metrics for that case.
#     Contains several rules for graph and time distances.

#     Parameters
#     --------------------------------------
#     graph: nx.DiGraph,
#         Process model graph
#     case: Case,
#         Case to compute the metrics
#     Returns
#     --------------------------------------
#     graph_distance, time_distance: Tuple[float, float]
#         Graph and time distances
#     """
#     trace = case.get_trace()
#     time = time_difference([case.get_timestamps()])[0]

#     if len(graph.edges) == 0 or len(trace) <= 1:
#         return 0, 0

#     # accumulates initial trace weight and graph times
#     graph_time = []
#     trace_weight = 0
#     for i in range(len(trace) - 1):
#         if (trace[i], trace[i + 1]) in graph.edges:
#             graph_time.append(graph[trace[i]][trace[i + 1]]["time_normalized"])
#             trace_weight += 1 - graph[trace[i]][trace[i + 1]]["weight_normalized"]
#         else:
#             graph_time.append(0)
#             trace_weight += 1

#     lent = len(trace) - 1
#     if lent > 1:
#         # trace has more than one edge (transition), graph distance is the division of the sum by the number of edges
#         graph_distance = trace_weight / lent
#     else:
#         # trace has only one edge, graph distance is the sum
#         graph_distance = trace_weight

#     difference = 0
#     for i in range(len(time)):
#         difference += abs(graph_time[i] - time[i])
#     if difference == 0:
#         # trace and graph have the exact same time distribution
#         return graph_distance, 0

#     graph_time_sum = sum(graph_time)
#     if graph_time_sum == 0:
#         # graph time sum is 0, returns the log of the difference
#         return graph_distance, log10(difference)
#     else:
#         # graph time sum and difference are not zero, original equation performed (most likely)
#         return graph_distance, log10(difference / graph_time_sum)
