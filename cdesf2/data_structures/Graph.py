from datetime import datetime as dt
from math import log10
from cdesf2.data_structures.Transition import Transition

"""
Generates and maintains the graphs and calculates GDtrace (gwd) and GDtime (twd).
"""

def timeProcessing(time_list):
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

def normGraph(graph):
    """
    Time and weight normalization for each Transition in the graph.
    Time normalization is the mean time.
    """
    max_weight = max([v._weight for v in graph.values()])
    for v in graph.values():
        v._weight_norm = v._weight / max_weight
        v._time_norm = v._time / v._count
    return graph

def createGraph(trace_list, time_list):
    """
    Creates a graph based on a list of traces and timestamps.
    """
    graph = {}
    time_list = timeProcessing(time_list)
    for trace, time in zip(trace_list, time_list):
        for i in range(len(trace)-1):
            path = (trace[i], trace[i+1])
            r_path = tuple(reversed(path))
            if path not in graph:
                graph[path] = Transition(path)
                graph[r_path] = Transition(r_path)
            graph[path].add(1, time[i])
            graph[r_path].add(1, time[i])
    return graph

def computeFeatures(graph, trace, raw_time):
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

    if len(graph.values()) == 0:
        return 0, 0
    graph = normGraph(graph)

    g_time = []
    trace_weight = 0
    for i in range(len(trace)-1):
        path = (trace[i], trace[i+1])
        if path in graph:
            g_time.append(graph[path]._time_norm)
            trace_weight += 1-graph[path]._weight_norm
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

def mergeGraphs(proc_graph, cp_graph):
    """
    Receives two graphs and merge them.
    Important to notice that the first graph has more weight
    and incorporates the second graph.
    5% of the original graph weight is disconted (decay).
    """
    for v in proc_graph.values():
        v._weight = 0.95 * v._weight

    for v in cp_graph.values():
        path = v._name
        if path in proc_graph:
            proc_graph[path]._weight += v._weight
            proc_graph[path]._time += v._time
            proc_graph[path]._count += v._count
        else:
            proc_graph[path] = Transition(path)
            proc_graph[path]._weight = v._weight
            proc_graph[path]._time = v._time
            proc_graph[path]._count = v._count
