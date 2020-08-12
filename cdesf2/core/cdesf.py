from typing import Union
from os import makedirs
from datetime import datetime
import networkx as nx
import pandas as pd
import numpy as np
from ..utils import extract_case_distances, initialize_graph, merge_graphs, Metrics
from ..data_structures import Case
from ..clustering import DenStream
from ..visualization import cumulative_stream_drifts, feature_space


class CDESF:
    """
    The CDESF class deals with the core concepts of CDESF.
    It controls several operations such processing new events, graphs management (Nyquist),
    DenStream triggering, plotting results and metrics recording.
    """

    def __init__(self,
                 name: str = 'Process',
                 time_horizon: int = 432000,
                 lambda_: float = 0.1,
                 beta: float = 0.2,
                 epsilon: float = 0.2,
                 mu: int = 4,
                 stream_speed: int = 100,
                 n_features: int = 2,
                 gen_plot: bool = False,
                 gen_metrics: bool = True):
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
        self.denstream = DenStream(lambda_, beta, epsilon, mu, stream_speed, n_features)
        self.cluster_metrics = []
        self.case_metrics = []
        self.active_core_clusters = set()
        self.drift_indexes = []
        self.metrics = Metrics(self.name)
        self.feature_space_plot_path = f'visualization/{self.name}_feature_space'
        makedirs(self.feature_space_plot_path, exist_ok=True)

    def get_case(self, case_id: str) -> Union[int, None]:
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
            graph_distance, time_distance = extract_case_distances(self.process_model_graph, case)
            case.graph_distance = graph_distance
            case.time_distance = time_distance

    def release_cases_from_memory(self) -> None:
        """
        Releases older cases based on the Nyquist value.
        The result is stored in self.cases.
        """
        self.cases = self.cases[:self.nyquist]

    def initialize_cdesf(self) -> None:
        """
        Initializes system variables, creates the first Process Model Graph and initializes DenStream.
        Initializes graphics and metrics functions.
        """
        self.nyquist = self.check_point_cases * 2
        # initialize PMG
        self.process_model_graph = initialize_graph(nx.DiGraph(), self.cases)
        # compute case metrics for initial cases
        self.initialize_case_metrics()

        # initialise denstream
        self.denstream.dbscan(self.cases)

        groups = self.denstream.generate_clusters()
        for group in groups[0]:
            for cluster in group:
                self.active_core_clusters.add(cluster.id)

        # plot
        if self.gen_plot:
            normals, outliers = [], []
            for case in self.cases:
                if not np.isnan(np.sum(case.point)) and self.denstream.is_normal(case.point):
                    normals.append(case.point)
                else:
                    outliers.append(case.point)
            feature_space(self.name,
                          self.event_index,
                          self.cp_count,
                          normals,
                          outliers,
                          self.denstream.generate_clusters(),
                          self.denstream.generate_outlier_clusters(),
                          self.denstream.epsilon,
                          self.feature_space_plot_path)

        # metrics
        if self.gen_metrics:
            for case in self.cases:
                self.metrics.compute_case_metrics(self.event_index, self.check_point, self.cp_count,
                                                  case, self.denstream.is_normal(case.point))
            self.metrics.save_case_metrics_on_check_point()
            self.metrics.compute_cluster_metrics(self.event_index, self.check_point, self.cp_count,
                                                 self.denstream.generate_clusters(),
                                                 self.denstream.generate_outlier_clusters())
            self.metrics.save_cluster_metrics_on_check_point()

        if len(self.process_model_graph.edges) > 0:
            self.metrics.save_pmg_on_check_point(self.process_model_graph, self.cp_count)

        self.initialized = True

    def set_case(self, case_id: str, act_name: str, act_timestamp: datetime) -> int:
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
        # check if case exists and creates one if it doesn't
        index = self.get_case(case_id)
        if index is None:
            self.cases.append(Case(case_id))
            self.check_point_cases += 1
            index = self.get_case(case_id)
        # add activity
        self.cases[index].set_activity(act_name, act_timestamp)
        # reorder list, putting the newest case in the first position
        self.cases.insert(0, self.cases.pop(index))

        return index

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

            check_point_graph = initialize_graph(nx.DiGraph(), self.cases)
            self.process_model_graph = merge_graphs(self.process_model_graph, check_point_graph)

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

    def process_event(self, case_id: str, act_name: str, act_timestamp: datetime) -> None:
        """
        The core function in the Process class.
        Sets a new case and controls the check point.
        If gp_creation is True, calculates the case metrics and uses
        them to train DenStream, recalculates the Nyquist and releases
        old cases if necessary.

        Parameters
        --------------------------------------
        case_id: str
            The case identifier
        act_name: str
            Activity's name
        act_timestamp: datetime
            Activity timestamp
        """
        self.total_cases.add(case_id)

        case_index = self.set_case(case_id, act_name, act_timestamp)
        current_time = self.cases[case_index].last_time

        if (current_time - self.check_point).total_seconds() > self.time_horizon and not self.initialized:
            """
            Initializes cdesf
            """
            self.check_point = current_time
            self.initialize_cdesf()
        elif self.initialized:
            """
            If we are past the first check point, graph distances are calculated and DenStream triggered
            """
            graph_distance, time_distance = extract_case_distances(self.process_model_graph, self.cases[case_index])
            self.cases[case_index].graph_distance = graph_distance
            self.cases[case_index].time_distance = time_distance

            # DENSTREAM
            self.denstream.train(self.cases[case_index])
            self.check_for_drift()

            # plots
            if self.gen_plot:
                normals, outliers = [], []
                for case in self.cases:
                    if not np.isnan(np.sum(case.point)) and self.denstream.is_normal(case.point):
                        normals.append(case.point)
                    else:
                        outliers.append(case.point)
                feature_space(self.name,
                              self.event_index,
                              self.cp_count,
                              normals,
                              outliers,
                              self.denstream.generate_clusters(),
                              self.denstream.generate_outlier_clusters(),
                              self.denstream.epsilon,
                              self.feature_space_plot_path)

            # metrics
            if self.gen_metrics:
                self.metrics.compute_case_metrics(self.event_index,
                                                  act_timestamp,
                                                  self.cp_count,
                                                  self.cases[case_index],
                                                  self.denstream.is_normal(self.cases[case_index].point))
                self.metrics.compute_cluster_metrics(self.event_index,
                                                     act_timestamp,
                                                     self.cp_count,
                                                     self.denstream.generate_clusters(),
                                                     self.denstream.generate_outlier_clusters())

            if (current_time - self.check_point).total_seconds() > self.time_horizon:
                """
                Check point
                """
                self.check_point = current_time
                self.check_point_update()

                self.metrics.save_pmg_on_check_point(self.process_model_graph, self.cp_count)
                if self.gen_metrics:
                    self.metrics.save_case_metrics_on_check_point()
                    self.metrics.save_cluster_metrics_on_check_point()

    def run(self, stream: np.ndarray) -> None:
        """
        Simulates the event stream by iterating through the stream variable
        generated reading the csv file, calls set_process at each new event

        Parameters
        --------------------------------------
        stream: pd.DataFrame
            Stream of events which was read by the read_csv function
        """
        for index, event in enumerate(stream):
            self.event_index = index
            case_id, activity_name, activity_timestamp = (event[0], event[1], event[2])
            if index == 0:
                self.check_point = activity_timestamp
            self.process_event(case_id, activity_name, activity_timestamp)

        self.drift_indexes = list(np.unique(self.drift_indexes))
        print("Total number of drifts:", len(self.drift_indexes))
        print("Drift points:", self.drift_indexes)
        cumulative_stream_drifts(len(stream), self.drift_indexes, f'visualization/drifts/{self.name}.pdf')
