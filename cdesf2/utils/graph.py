from datetime import datetime as dt
from math import log10
import networkx as nx

"""
Generates and maintains the graphs and calculates GDtrace (gwd) and GDtime (twd).
"""


def time_processing(time_list):
    """
    This function receives a list of lists containing timestamps
    from the selected cases.
    It processes the timestamps and calculates the time difference
    within activities for all separate cases.
    """
    outer_time = []
    for time in time_list:
        inner_time = []
        for i in range(len(time)-1):
            time_diff = ((dt.strptime(time[i+1], '%Y/%m/%d %H:%M:%S.%f'))
                         - (dt.strptime(time[i], '%Y/%m/%d %H:%M:%S.%f'))
                         ).total_seconds()
            inner_time.append(time_diff)
        if len(inner_time) == 0:
            inner_time.append(0)
        outer_time.append(inner_time)
    return outer_time


def norm_graph(graph):
    """
    Time and weight normalization for each Transition in the graph.
    Time normalization is the mean time.
    """
    max_weight = max([data['weight'] for n1, n2, data in graph.edges(data=True)])
    for n1, n2, data in graph.edges(data=True):
        data['weight_norm'] = data['weight'] / max_weight
        data['time_norm'] = data['time'] / data['count']
    return graph


def create_graph(trace_list, time_list):
    """
    Creates a graph based on a list of traces and timestamps.
    """
    graph = nx.Graph()
    time_list = time_processing(time_list)
    for trace, s_time in zip(trace_list, time_list):
        for i in range(len(trace)-1):
            path = (trace[i], trace[i+1])
            if path not in graph.edges:
                graph.add_edge(*path, weight=0, time=0,
                               weight_norm=0, time_norm=0, count=0)
            graph[path[0]][path[1]]['weight'] += 1
            graph[path[0]][path[1]]['time'] += s_time[i]
            graph[path[0]][path[1]]['count'] += 1
    return graph


def compute_features(graph, trace, raw_time):
    """
    Receives a graph, trace and timestamps from a selected case and
    computes the metrics for that case.
    Contains several rules for GDtrace (gwd) and GDtime (twd).
    """
    time = []
    for i in range(len(raw_time)-1):
        time_diff = ((dt.strptime(raw_time[i+1], '%Y/%m/%d %H:%M:%S.%f'))
                     - (dt.strptime(raw_time[i], '%Y/%m/%d %H:%M:%S.%f'))
                     ).total_seconds()
        time.append(time_diff)

    if len(graph.edges) == 0:
        return 0, 0
    graph = norm_graph(graph)

    g_time = []
    trace_weight = 0
    for i in range(len(trace)-1):
        if (trace[i], trace[i+1]) in graph.edges:
            g_time.append(graph[trace[i]][trace[i+1]]['time_norm'])
            trace_weight += 1-graph[trace[i]][trace[i+1]]['weight_norm']
        else:
            g_time.append(0)
            trace_weight += 1

    lent = len(trace)-1
    if lent > 1:
        gwd = trace_weight / lent
    else:
        gwd = trace_weight

    dif = 0
    for i in range(len(time)):
        dif += abs(g_time[i] - time[i])

    g_sum = sum(g_time)
    if g_sum == 0:
        if dif == 0:
            twd = 0
        else:
            twd = log10(dif)
    elif dif == 0:
        twd = 0
    else:
        twd = log10((dif/g_sum))

    return gwd, twd


def merge_graphs(proc_graph, cp_graph):
    """
    Receives two graphs and merge them.
    Important to notice that the first graph has more weight
    and incorporates the second graph.
    5% of the original graph weight is disconted (decay).
    """
    for n1, n2, data in proc_graph.edges(data=True):
        data['weight'] = 0.95 * data['weight']

    for n1, n2, data in cp_graph.edges(data=True):
        path = (n1, n2)
        if path in proc_graph.edges:
            proc_graph[n1][n2]['weight'] += data['weight']
            proc_graph[n1][n2]['time'] += data['time']
            proc_graph[n1][n2]['count'] += data['count']
        else:
            proc_graph.add_edge(*path, weight=0, time=0,
                                weight_norm=0, time_norm=0, count=0)
            proc_graph[n1][n2]['weight'] += data['weight']
            proc_graph[n1][n2]['time'] += data['time']
            proc_graph[n1][n2]['count'] += data['count']
