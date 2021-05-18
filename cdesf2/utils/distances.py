import networkx as nx
import collections
from numpy import average, count_nonzero
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
        is_numerical = True
        current_total = 0
        difference_total = 0

        for index, activity in enumerate(trace):
            case_value = case.get_attribute(attribute_name)[index]

            activity_node: dict = graph.nodes.get(activity)

            if not activity_node:
                # This is the first time we see this activity, so the distance is maximum.
                # This might be the first ever event for this attribute, so we also need to check to see if it is
                # numerical or categorical
                if isinstance(case_value, str):
                    current_total += 1
                else:
                    current_total += 0  # The average for this attribute so far should remain the same
                    difference_total += case_value  # The distance should be maximum

                continue

            activity_attribute_values = activity_node.get(attribute_name)

            if isinstance(case_value, str):
                # Handle this attribute like a categorical attribute
                is_numerical = False
                activity_attribute_counter = collections.Counter(
                    activity_attribute_values
                )
                activity_attribute_frequency = activity_attribute_counter.get(
                    case_value, 0
                )
                activity_attribute_distance = 1 - (
                    activity_attribute_frequency / len(activity_attribute_values)
                )

                current_total += activity_attribute_distance
            else:
                # Handle this attribute like a numerical attribute
                is_numerical = True
                activity_attribute_average = average(activity_attribute_values)
                current_total += activity_attribute_average

                difference_total += abs(activity_attribute_average - case_value)

        if is_numerical:
            distances[attribute_name] = difference_total / current_total
        else:
            distances[attribute_name] = current_total / len(trace)

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
