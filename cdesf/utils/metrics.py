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

    case_columns: "list[str]"
    cluster_columns: "list[str]"
    additional_attributes: "list[str]"

    def __init__(self, file_name: str, additional_attributes: "list[str]" = []):
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
        self.additional_attributes = additional_attributes

        self.path_to_pmg_metrics = f"output/metrics/{file_name}_process_model_graphs"
        self.path_to_pmg_vis = f"output/visualization/{file_name}_process_model_graphs"
        self.path_to_drifts = "output/visualization/drifts"
        self.path_to_case_metrics = "output/metrics/case_metrics"
        self.path_to_cluster_metrics = "output/metrics/cluster_metrics"
        try:
            makedirs(self.path_to_pmg_metrics, exist_ok=True)
            makedirs(self.path_to_pmg_vis, exist_ok=True)
            makedirs(self.path_to_drifts, exist_ok=True)
            makedirs(self.path_to_case_metrics, exist_ok=True)
            makedirs(self.path_to_cluster_metrics, exist_ok=True)

            self.case_columns, self.cluster_columns = self.generate_column_names()

            pd.DataFrame(columns=self.case_columns).to_csv(
                f"{self.path_to_case_metrics}/{file_name}.csv", index=False
            )
            pd.DataFrame(columns=self.cluster_columns).to_csv(
                f"{self.path_to_cluster_metrics}/{file_name}.csv", index=False
            )
        except Exception as e:
            print(e)

    def generate_column_names(self) -> Tuple["list[str]", "list[str]"]:
        case_columns = []
        cluster_columns = []

        case_columns += [
            "stream_index",
            "timestamp",
            "check point",
            "case",
            "graph distance",
            "time distance",
        ]
        case_columns += [
            f"{attribute_name} distance"
            for attribute_name in self.additional_attributes
        ]
        case_columns += ["label"]

        cluster_columns += [
            "stream_index",
            "timestamp",
            "check point",
            "cluster id",
            "graph coordinate",
            "time coordinate",
        ]
        cluster_columns += [
            f"{attribute_name} coordinate"
            for attribute_name in self.additional_attributes
        ]
        cluster_columns += ["radius", "weight", "cluster type"]

        return case_columns, cluster_columns

    def compute_case_metrics(
        self,
        event_index: int,
        timestamp: datetime,
        cp_count: int,
        case: Case,
        label: bool,
    ) -> None:
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
        label_str = "anomalous"
        if label:
            label_str = "normal"

        data = [
            event_index,
            timestamp,
            cp_count,
            case.id,
            case.distances.get("graph"),
            case.distances.get("time"),
        ]

        for attribute_name in self.additional_attributes:
            data.append(case.distances.get(attribute_name))

        # `label` is always the last column
        data.append(label_str)

        self.case_metrics.append(data)

    def save_case_metrics_on_check_point(self) -> None:
        """
        Saves the case metrics into a file according to a set path and name.
        Also releases the case_metrics attribute
        """
        cm_path = f"{self.path_to_case_metrics}/{self.file_name}.csv"
        pd.read_csv(cm_path).append(
            pd.DataFrame(self.case_metrics, columns=self.case_columns)
        ).to_csv(cm_path, index=False)
        self.case_metrics.clear()

    def compute_cluster_metrics_helper(
        self,
        event_index: int,
        timestamp: datetime,
        cp_count: int,
        cluster: Cluster,
        cluster_type: str,
    ) -> None:
        """
        Helper function to save metrics into cluster_metrics attribute.
        """
        data = [
            event_index,
            timestamp,
            cp_count,
            cluster.id,
        ]

        for dimension in cluster.centroid:
            data.append(dimension)

        data += [
            cluster.radius,
            cluster.weight,
            cluster_type,
        ]

        self.cluster_metrics.append(data)

    def compute_cluster_metrics(
        self,
        event_index: int,
        timestamp: datetime,
        cp_count: int,
        normal_clusters: Tuple[List[List], List[List]],
        o_clusters: List[Cluster],
    ) -> None:
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
                self.compute_cluster_metrics_helper(
                    event_index, timestamp, cp_count, cluster, "core micro-cluster"
                )

        for group in p_clusters:
            for cluster in group:
                self.compute_cluster_metrics_helper(
                    event_index, timestamp, cp_count, cluster, "potential micro-cluster"
                )

        for cluster in o_clusters:
            self.compute_cluster_metrics_helper(
                event_index, timestamp, cp_count, cluster, "outlier micro-cluster"
            )

    def save_cluster_metrics_on_check_point(self) -> None:
        """
        Saves the cluster metrics into a file according to a set path and name.
        Also releases the cluster_metrics attribute
        """
        cm_path = f"{self.path_to_cluster_metrics}/{self.file_name}.csv"
        pd.read_csv(cm_path).append(
            pd.DataFrame(self.cluster_metrics, columns=self.cluster_columns)
        ).to_csv(cm_path, index=False)
        self.cluster_metrics.clear()

    def save_pmg_on_check_point(
        self, process_model_graph: nx.DiGraph, cp_count: int
    ) -> None:
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
            with open(
                f"{self.path_to_pmg_metrics}/process_model_graph_{cp_count}.json", "w"
            ) as file:
                file.write(
                    json.dumps(
                        nx.readwrite.json_graph.node_link_data(process_model_graph)
                    )
                )

            save_graphviz(
                process_model_graph,
                f"{self.path_to_pmg_vis}/process_model_graph_{cp_count}",
            )
        except Exception as e:
            print(e)
