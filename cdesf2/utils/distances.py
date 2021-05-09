import networkx as nx
from numpy import average
from math import log10
from ..data_structures import Case


def calculate_case_distances(
    graph: nx.DiGraph, case: Case, *, additional_attributes: "list[str]" = []
) -> "dict[str, float]":
    """
    Extracts the distances between the current graph and a given case across trace,
    time, and the additional attributes provided.

    Arguments:
        graph: nx.DiGraph

        case: Case

        additional_attributes: list[str]:
            A list of additional attributes to calculate the distances for.

    Returns:
        A dictionary that associates each attribute with the distance between the
        graph and the case.
    """
    trace = case.get_trace()
    time_differences = calculate_case_time_distances(case)

    if not graph.edges or len(trace) < 2:
        # Return a map with all-zero distances.
        default_distances = {"graph": 0.0, "time": 0.0}

        for attribute_name in additional_attributes:
            default_distances[attribute_name] = 0.0

        return default_distances

    distances: "dict[str, float]" = {}

    # Node distances
    for attribute_name in additional_attributes:
        current_total = 0
        difference_total = 0

        for index, activity in enumerate(trace):
            activity_attribute_values = graph.nodes[activity][attribute_name]
            activity_attribute_average = average(activity_attribute_values)
            current_total += activity_attribute_average

            case_value = case.get_attribute(attribute_name)[index]
            difference_total += abs(activity_attribute_average - case_value)

        distances[attribute_name] = difference_total / current_total

    # Edge distances
    normalized_times: "list[float]" = []
    normalized_weight = 0

    for trace_index in range(len(trace) - 1):
        first_node = trace[trace_index]
        second_node = trace[trace_index + 1]

        if graph.has_edge(first_node, second_node):
            edge = graph[first_node][second_node]
            normalized_times.append(edge["time_normalized"])
            normalized_weight += 1 - edge["weight_normalized"]
        else:
            normalized_times.append(0.0)
            normalized_weight += 1

    distances["graph"] = normalized_weight / (len(trace) - 1)

    time_current_total = sum(normalized_times)
    time_difference_total = 0

    for time_index, normalized_time in enumerate(normalized_times):
        time_difference_total += abs(normalized_time - time_differences[time_index])

    if time_difference_total == 0:
        distances["time"] = 0.0
    elif time_current_total == 0:
        distances["time"] = log10(time_difference_total)
    else:
        distances["time"] = log10(time_difference_total / time_current_total)

    return distances


def calculate_case_time_distances(case: Case) -> "list[float]":
    timestamps = case.get_timestamps()

    if len(timestamps) < 2:
        return [0]

    distances: "list[float]" = []

    for lhs, rhs in list(zip(timestamps, timestamps[1:])):
        distances.append((rhs - lhs).total_seconds())

    return distances
