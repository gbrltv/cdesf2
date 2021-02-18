from typing import List, Tuple
import pandas as pd
from datetime import datetime
from os import makedirs
import json
import networkx as nx
from ..data_structures import Case, Cluster
from ..visualization import save_graphviz


class Metrics:
    """
    Controls the computation of metrics during the stream processing
    and writes the results into files periodically (check points)
    """

    def __init__(self, file_name: str):
        """
        Creates the paths for the outputs and initializes the metrics attributes

        Parameters
        --------------------------------------
        file_name: str
            Process name, used for the path creation
        """
        self.case_metrics = []
        self.cluster_metrics = []
        self.file_name = file_name

        self.path_to_pmg_metrics = f'output/metrics/{file_name}_process_model_graphs'
        self.path_to_pmg_vis = f'output/visualization/{file_name}_process_model_graphs'
        self.path_to_drifts = 'output/visualization/drifts'
        self.path_to_case_metrics = 'output/metrics/case_metrics'
        self.path_to_cluster_metrics = 'output/metrics/cluster_metrics'
        try:
            makedirs(self.path_to_pmg_metrics, exist_ok=True)
            makedirs(self.path_to_pmg_vis, exist_ok=True)
            makedirs(self.path_to_drifts, exist_ok=True)
            makedirs(self.path_to_case_metrics, exist_ok=True)
            makedirs(self.path_to_cluster_metrics, exist_ok=True)

            pd.DataFrame(columns=['stream_index', 'timestamp', 'check point', 'case',
                                  'graph distance', 'time distance', 'label']) \
                .to_csv(f'{self.path_to_case_metrics}/{file_name}.csv', index=False)
            pd.DataFrame(columns=['stream_index', 'timestamp', 'check point', 'cluster id',
                                  'x', 'y', 'radius', 'weight', 'cluster type']) \
                .to_csv(f'{self.path_to_cluster_metrics}/{file_name}.csv', index=False)
        except Exception as e:
            print(e)

    def compute_case_metrics(self, event_index: int, timestamp: datetime, cp_count: int, case: Case, label: bool) -> None:
        """
        Generates case metrics and saves them in the self.case_metrics attribute.

        Parameters
        --------------------------------------
        event_index: int
            Index of the current event
        timestamp: datetime
            Current timestamp of the stream
        cp_count: int
            Current check point
        case: Case
            Case to be saved
        label: bool
            Controls if case is normal or anomalous
        """
        label_str = 'anomalous'
        if label:
            label_str = 'normal'

        self.case_metrics.append([event_index, timestamp, cp_count, case.id,
                                  case.graph_distance, case.time_distance, label_str])

    def save_case_metrics_on_check_point(self) -> None:
        """
        Saves the case metrics into a file according to a set path and name.
        Also releases the case_metrics attribute
        """
        cm_path = f'{self.path_to_case_metrics}/{self.file_name}.csv'
        columns = ['stream_index', 'timestamp', 'check point',
                   'case', 'graph distance', 'time distance', 'label']
        pd.read_csv(cm_path).append(pd.DataFrame(self.case_metrics,
                                                 columns=columns)).to_csv(cm_path, index=False)
        self.case_metrics.clear()

    def compute_cluster_metrics_helper(self, event_index: int, timestamp: datetime, cp_count: int,
                                       cluster: Cluster, cluster_type: str) -> None:
        """
        Helper function to save metrics into cluster_metrics attribute.
        """
        self.cluster_metrics.append([event_index, timestamp, cp_count, cluster.id, cluster.centroid[0],
                                     cluster.centroid[1], cluster.radius, cluster.weight, cluster_type])

    def compute_cluster_metrics(self, event_index: int, timestamp: datetime, cp_count: int,
                                normal_clusters: Tuple[List[List], List[List]], o_clusters: List[Cluster]) -> None:
        """
        Generates cluster metrics and saves them in the self._cluster_metrics attribute.

        Parameters
        --------------------------------------
        event_index: int
            Index of the current event
        timestamp: datetime
            Current timestamp of the stream
        cp_count: int
            Current check point
        normal_clusters: Tuple[List[List], List[List]]
            Core and potential micro-clusters maintained by denstream
        o_clusters: List[Cluster]
            Outlier micro-clusters maintained by denstream
        """
        c_clusters, p_clusters = normal_clusters[0], normal_clusters[1]
        for group in c_clusters:
            for cluster in group:
                self.compute_cluster_metrics_helper(event_index, timestamp, cp_count, cluster, 'core micro-cluster')

        for group in p_clusters:
            for cluster in group:
                self.compute_cluster_metrics_helper(event_index, timestamp, cp_count, cluster, 'potential micro-cluster')

        for cluster in o_clusters:
            self.compute_cluster_metrics_helper(event_index, timestamp, cp_count, cluster, 'outlier micro-cluster')

    def save_cluster_metrics_on_check_point(self) -> None:
        """
        Saves the cluster metrics into a file according to a set path and name.
        Also releases the cluster_metrics attribute
        """
        cm_path = f'{self.path_to_cluster_metrics}/{self.file_name}.csv'
        columns = ['stream_index', 'timestamp', 'check point', 'cluster id',
                   'x', 'y', 'radius', 'weight', 'cluster type']
        pd.read_csv(cm_path).append(pd.DataFrame(
            self.cluster_metrics, columns=columns)).to_csv(cm_path, index=False)
        self.cluster_metrics.clear()

    def save_pmg_on_check_point(self, process_model_graph: nx.DiGraph, cp_count: int) -> None:
        """
        Saves the Process Model Graph at all check points in a JSON file and plots

        Parameters
        --------------------------------------
        process_model_graph: nx.DiGraph
            Process model graph on the current check point
        cp_count: int
            Current check point
        """
        try:
            with open(f'{self.path_to_pmg_metrics}/process_model_graph_{cp_count}.json', 'w') as file:
                file.write(json.dumps(nx.readwrite.json_graph.node_link_data(process_model_graph)))

            save_graphviz(process_model_graph, f'{self.path_to_pmg_vis}/process_model_graph_{cp_count}')
        except Exception as e:
            print(e)
