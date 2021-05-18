import multiprocessing as mp
from typing import Union
from os import makedirs
from datetime import datetime
import networkx as nx
import pandas as pd
import numpy as np
from ..utils import calculate_case_distances, initialize_graph, merge_graphs, Metrics
from ..data_structures import Case
from ..clustering import DenStream
from ..visualization import cumulative_stream_drifts, feature_space
from pm4py.objects.log import obj


class CDESF:
    """
    The CDESF class deals with the core concepts of CDESF.
    It controls several operations such processing new events, graphs management (Nyquist),
    DenStream triggering, plotting results and metrics recording.
    """

    cases: "list[Case]"
    additional_attributes: "list[str]"

    def __init__(
        self,
        name: str = "Process",
        time_horizon: int = 432000,
        lambda_: float = 0.1,
        beta: float = 0.2,
        epsilon: float = 0.2,
        mu: int = 4,
        stream_speed: int = 100,
        gen_plot: bool = False,
        gen_metrics: bool = True,
        additional_attributes: "list[str]" = [],
    ):
        """
        This function sets up a new process, defining its name, and preparing initial attributes.

        Parameters
        --------------------------------------
        name: str
            Name of the process
        time_horizon: int
            The time horizon window that controls Check Point occurence
            (unit: seconds)
        lambda_: float
            Sets the importance of historical data for
            the current clusters
        beta: float
            Controls micro-cluster weights and promotion
        epsilon: float
            Defines the maximum range of a micro-cluster action
        mu: int
            Controls micro-cluster weights and promotion
        stream_speed: int
            Controls how frequent the decay factor (lambda)
            influences the micro-clusters
        n_features: int
            The number of features DenStream must consider,
            in our case is always set to 2, since we have
            two attributes (graph_distance and time_distance)
        gen_plot: bool
            If True, triggers the plot
            generation
        gen_metrics: bool
            If True, triggers the
            metrics generation
        """
        self.gen_plot = gen_plot
        self.gen_metrics = gen_metrics
        self.event_index = 0
        self.total_cases = set()
        self.check_point = datetime(2010, 1, 1)
        self.cases = []
        self.name = name
        self.time_horizon = time_horizon
        self.initialized = False
        self.cp_count = 0
        self.nyquist = 0
        self.check_point_cases = 0
        self.process_model_graph = nx.DiGraph()
        self.additional_attributes = additional_attributes
        self.n_features = len(additional_attributes) + 2
        self.denstream = DenStream(
            lambda_, beta, epsilon, mu, stream_speed, self.n_features
        )
        self.cluster_metrics = []
        self.case_metrics = []
        self.active_core_clusters = set()
        self.drift_indexes = []
        self.metrics = Metrics(self.name, additional_attributes)
        self.feature_space_plot_path = f"output/visualization/{self.name}_feature_space"

        # Used to store the current feature_space jobs that are executing.
        self.fs_pool = mp.Pool(mp.cpu_count())

        makedirs(self.feature_space_plot_path, exist_ok=True)

    def get_case_index(self, case_id: str) -> Union[int, None]:
        """
        Returns the case index from the cases list.

        Parameters
        --------------------------------------
        case_id: str
            The case identifier

        Returns
        --------------------------------------
        The case position in the case list
        """
        for index, case in enumerate(self.cases):
            if case.id == case_id:
                return index
        return None

    def initialize_case_metrics(self) -> None:
        """
        Calculates GDtrace (graph_distance) and GDtime (time_distance)
        for cases in the first time horizon cycle.
        Initializes metrics in case the user triggers this task
        """
        for case in self.cases:
            case.distances = calculate_case_distances(
                self.process_model_graph,
                case,
                additional_attributes=self.additional_attributes,
            )

    def release_cases_from_memory(self) -> None:
        """
        Releases older cases based on the Nyquist value.
        The result is stored in self.cases.
        """
        self.cases = self.cases[: self.nyquist]

    def initialize_cdesf(self) -> None:
        """
        Initializes system variables, creates the first Process Model Graph and initializes DenStream.
        Initializes graphics and metrics functions.
        """
        self.nyquist = self.check_point_cases * 2
        # initialize PMG
        self.process_model_graph = initialize_graph(
            self.cases, self.additional_attributes
        )
        # compute case metrics for initial cases
        self.initialize_case_metrics()

        # initialise denstream
        self.denstream.dbscan(self.cases)

        groups = self.denstream.generate_clusters()
        for group in groups[0]:
            for cluster in group:
                self.active_core_clusters.add(cluster.id)

        # Computed outside mainly to ensure that the value of the clusters
        # gets calculated sequentially and not inside the pool.
        generated_o_clusters = self.denstream.generate_outlier_clusters()

        # Plot
        # Plotting is only available when no additional attributes are defined
        if self.gen_plot and not len(self.additional_attributes):
            normals, outliers = [], []
            for case in self.cases:
                if not np.isnan(np.sum(case.point)) and self.denstream.is_normal(
                    case.point
                ):
                    normals.append(case.point)
                else:
                    outliers.append(case.point)

            self.fs_pool.apply_async(
                func=feature_space,
                args=(
                    self.name,
                    self.event_index,
                    self.cp_count,
                    normals,
                    outliers,
                    groups,
                    generated_o_clusters,
                    self.denstream.epsilon,
                    self.feature_space_plot_path,
                ),
            )

        # metrics
        if self.gen_metrics:
            for case in self.cases:
                self.metrics.compute_case_metrics(
                    self.event_index,
                    self.check_point,
                    self.cp_count,
                    case,
                    self.denstream.is_normal(case.point),
                )
            self.metrics.save_case_metrics_on_check_point()
            self.metrics.compute_cluster_metrics(
                self.event_index,
                self.check_point,
                self.cp_count,
                self.denstream.generate_clusters(),
                self.denstream.generate_outlier_clusters(),
            )
            self.metrics.save_cluster_metrics_on_check_point()

        if len(self.process_model_graph.edges) > 0:
            self.metrics.save_pmg_on_check_point(
                self.process_model_graph, self.cp_count
            )

        self.initialized = True

    def set_case(self, case_id: str, event) -> int:
        """
        Function that finds the case and if it already exists,
        sets the activity with the name and the timestamp passed and
        puts the case modified in the first position in the list of cases.
        If it does not exist, it creates a new case and adds the activity.
        It returns the index of the case modified/created

        Parameters
        --------------------------------------
        case_id: str
            The case identifier,
        act_name: str
            Activity's name,
        act_timestamp: datetime
            Activity timestamp
        Returns
        --------------------------------------
        index: int
            Index of the case modified/created
        """
        case_index = self.get_case_index(case_id)

        if case_index is None:
            new_case = Case(case_id)
            self.cases.append(new_case)
            self.check_point_cases += 1
            case_index = len(self.cases) - 1

        self.cases[case_index].add_event(event)

        # reorder list, putting the newest case in the first position
        self.cases.insert(0, self.cases.pop(case_index))

        return case_index

    def check_point_update(self):
        """
        Check point is reached. It means that the process model graph
        is updated and older cases are released from memory
        """
        self.cp_count += 1

        if len(self.cases) > self.nyquist:
            """
            Recalculates nyquist, releases cases and updates model (merges graphs)
            """
            self.release_cases_from_memory()
            if self.check_point_cases > 5:
                self.nyquist = self.check_point_cases * 2

            check_point_graph = initialize_graph(self.cases, self.additional_attributes)
            self.process_model_graph = merge_graphs(
                self.process_model_graph, check_point_graph
            )

        self.check_point_cases = 0

    def check_for_drift(self):
        """
        Detects if a drift has occurred for each new event in the stream.
        If a new core behavior (represented by c-micro-clusters) appears,
        then a drift alert is triggered.
        """
        old_clusters = self.active_core_clusters.copy()
        groups = self.denstream.generate_clusters()
        for group in groups[0]:
            for cluster in group:
                if cluster.id not in self.active_core_clusters:
                    self.active_core_clusters.add(cluster.id)
                    self.drift_indexes.append(self.event_index)
                    print("DRIFT ALERT")
                    print("Stream position:", self.event_index)
                    print("New core behavior detected: cluster", cluster.id)
                    print("Cluster weight:", cluster.weight)
                    print("Cluster radius:", cluster.radius)
                    print("Cases in cluster:", cluster.case_ids)
                    print()
                else:
                    old_clusters.remove(cluster.id)

        # for cluster in old_clusters:
        #     self.active_core_clusters.remove(cluster)
        #     self.drift_indexes.append(event_index)
        #     print("DRIFT ALERT")
        #     print("Cluster", cluster, "ceased to exist")
        #     print()

    def process_event(self, event):
        case_id = event["case:concept:name"]
        case_index = self.set_case(case_id, event)
        case = self.cases[case_index]

        self.total_cases.add(case_id)

        current_time = case.last_time
        time_distance = (current_time - self.check_point).total_seconds()

        if not self.initialized:
            if time_distance > self.time_horizon:
                self.check_point = current_time
                self.initialize_cdesf()
            return

        case.distances = calculate_case_distances(
            self.process_model_graph,
            case,
            additional_attributes=self.additional_attributes,
        )

        # DenStream
        self.denstream.train(case)
        self.check_for_drift()

        # Computed outside mainly to ensure that the value of the clusters
        # gets calculated sequentially and not inside the pool.
        generated_clusters = self.denstream.generate_clusters()
        generated_o_clusters = self.denstream.generate_outlier_clusters()

        # Plots
        if self.gen_plot:
            normals, outliers = [], []
            for case in self.cases:
                if not np.isnan(np.sum(case.point)) and self.denstream.is_normal(
                    case.point
                ):
                    normals.append(case.point)
                else:
                    outliers.append(case.point)

            self.fs_pool.apply_async(
                func=feature_space,
                args=(
                    self.name,
                    self.event_index,
                    self.cp_count,
                    normals,
                    outliers,
                    generated_clusters,
                    generated_o_clusters,
                    self.denstream.epsilon,
                    self.feature_space_plot_path,
                ),
            )

        # Metrics
        if self.gen_metrics:
            self.metrics.compute_case_metrics(
                self.event_index,
                event.get("time:timestamp"),
                self.cp_count,
                self.cases[case_index],
                self.denstream.is_normal(self.cases[case_index].point),
            )
            self.metrics.compute_cluster_metrics(
                self.event_index,
                event.get("time:timestamp"),
                self.cp_count,
                self.denstream.generate_clusters(),
                self.denstream.generate_outlier_clusters(),
            )

        if time_distance > self.time_horizon:
            self.check_point = current_time
            self.check_point_update()
            self.metrics.save_pmg_on_check_point(
                self.process_model_graph, self.cp_count
            )
            if self.gen_metrics:
                self.metrics.save_case_metrics_on_check_point()
                self.metrics.save_cluster_metrics_on_check_point()

    def run(self, stream: obj.EventStream) -> None:
        """
        Simulates the event stream by iterating through the stream variable
        generated reading the input file, calls set_process at each new event

        Parameters
        --------------------------------------
        stream: obj.EventStream
            Event stream imported by PM4Py
        """

        for index, event in enumerate(stream):
            self.event_index = index

            if index == 0:
                self.check_point = event["time:timestamp"]

            self.process_event(event)
            # self.event_index = index
            # case_id, activity_name, activity_timestamp = (
            #     event["case:concept:name"],
            #     event["concept:name"],
            #     event["time:timestamp"],
            # )
            # if index == 0:
            #     self.check_point = activity_timestamp
            # self.process_event(case_id, activity_name, activity_timestamp)

        self.fs_pool.close()
        self.fs_pool.join()

        self.drift_indexes = list(np.unique(self.drift_indexes))

        print("Total number of drifts:", len(self.drift_indexes))
        print("Drift points:", self.drift_indexes)

        cumulative_stream_drifts(
            len(stream),
            self.drift_indexes,
            f"output/visualization/drifts/{self.name}.pdf",
        )
