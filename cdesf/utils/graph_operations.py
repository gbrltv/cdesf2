import networkx as nx
from ..data_structures.case import Case
from .distances import calculate_case_time_distances


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
    edges = graph.edges(data=True)
    attributes: "list[dict[str, float]]" = [attributes for _, _, attributes in edges]

    weights = map(lambda attribute: attribute.get("weight"), attributes)
    max_weight = max(weights)

    for data in attributes:
        edge_weight = data.get("weight")
        edge_time = data.get("time")

        data["weight_normalized"] = edge_weight / max_weight
        data["time_normalized"] = edge_time / edge_weight

    return graph


def initialize_graph(
    case_list: "list[Case]", additional_attributes: "list[str]" = []
) -> nx.DiGraph:
    """
    Initialize a graph based on the weights and time differences from a list of cases

    Parameters
    --------------------------------------
    case_list: list[Case]
        List of cases used to initialize the graph

    Returns
    --------------------------------------
    graph: nx.DiGraph,
        Initialized graph
    """
    graph = nx.DiGraph()

    for case in case_list:
        trace = case.get_trace()
        time_differences = calculate_case_time_distances(case)

        # TODO: This code can be optimized heavily to not have this many for-loops.
        for index, activity in enumerate(trace):
            if activity not in graph.nodes:
                attributes = {}
                for attribute_name in additional_attributes:
                    attributes[attribute_name] = [
                        case.get_attribute(attribute_name)[index]
                    ]

                graph.add_node(activity, **attributes)
            else:
                for attribute_name in additional_attributes:
                    graph.nodes[activity][attribute_name].append(
                        case.get_attribute(attribute_name)[index]
                    )

        for trace_index in range(len(trace) - 1):
            edge = (trace[trace_index], trace[trace_index + 1])

            if edge not in graph.edges:
                graph.add_edge(*edge, weight=1, time=time_differences[trace_index])
            else:
                graph[edge[0]][edge[1]]["weight"] += 1
                graph[edge[0]][edge[1]]["time"] += time_differences[trace_index]

    return normalize_graph(graph)


def merge_graphs(
    process_model_graph: nx.DiGraph, check_point_graph: nx.DiGraph
) -> nx.DiGraph:
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
        data["weight"] *= 0.95

    for node1, node2, data in check_point_graph.edges(data=True):
        path = (node1, node2)
        if path in process_model_graph.edges:
            process_model_graph[node1][node2]["weight"] += data["weight"]
            process_model_graph[node1][node2]["time"] += data["time"]
        else:
            process_model_graph.add_edge(
                *path, weight=data["weight"], time=data["time"]
            )

    for activity, data in check_point_graph.nodes(data=True):
        if activity not in process_model_graph.nodes:
            process_model_graph.add_node(activity, **data)
        else:
            for attribute in process_model_graph.nodes[activity]:
                process_model_graph.nodes[activity][attribute].extend(
                    check_point_graph.nodes[activity][attribute]
                )

    return normalize_graph(process_model_graph)
