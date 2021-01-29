import shutil
from os import path

import networkx as nx
import pytest
from cdesf2.visualization import (cumulative_stream_drifts, save_graph,
                                  save_graphviz)


def test_save_graph():
    graph = nx.DiGraph()

    graph.add_edge('activity a', 'b', weight=0.6)
    graph.add_edge('activity a', 'c', weight=0.2)
    graph.add_edge('c', 'd', weight=0.1)
    graph.add_edge('c', 'e', weight=0.7)
    graph.add_edge('c', 'f', weight=0.9)
    graph.add_edge('activity a', 'd', weight=0.3)

    save_graph(graph, 'output/graphs/graph.png')
    assert path.isfile('output/graphs/graph.png')


def test_save_graphviz():
    graph = nx.DiGraph()

    graph.add_edge('activity a', 'b', weight=0.6)
    graph.add_edge('activity a', 'c', weight=0.2)
    graph.add_edge('c', 'd', weight=0.1)
    graph.add_edge('c', 'e', weight=0.7)
    graph.add_edge('c', 'f', weight=0.9)
    graph.add_edge('activity a', 'd', weight=0.3)

    save_graphviz(graph, 'output/graphs/graphviz.png')
    assert path.isfile('output/graphs/graphviz.png')


def test_cummulative_stream_drifts():
    stream_length = 5000
    drifts_index = [22, 26, 28, 149, 153, 161, 172, 187, 260, 1117, 1164, 1199,
                    1265, 1359, 1364, 1369, 1508, 1517, 1518, 1567, 1690, 1779, 2083]

    cumulative_stream_drifts(stream_length, drifts_index,
                             'output/test_cumulative_stream_drifts.pdf')
    assert path.isfile('output/test_cumulative_stream_drifts.pdf')
