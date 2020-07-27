from typing import Union
from datetime import datetime
import networkx as nx
import pandas as pd
from ..utils import extract_case_distances, initialize_graph, normalize_graph, merge_graphs
from ..data_structures import Case
from ..clustering import DenStream
# from visualization import gen_data_plot, plot_clusters, cluster_metrics


class Process:
    """
    The Process class deals with the core concepts of CDESF controlling several
    operations such as the setting of a case, graphs management (Nyquist),
    DenStream triggering, plotting results and metrics recording.
    """

    def __init__(self, name: str, timestamp: datetime, time_horizon: int, gen_plot: bool, plot_path: str,
                 gen_metrics: bool, metrics_path: str, denstream_kwargs: dict):
        """
        This function sets up a new process, defining its name,
        and preparing initial attributes.

        Parameters
        --------------------------------------
        name: str
            Name of the process,
        timestamp: datetime
            Timestamp for the first event
            which will be used as a mark
            for the first check,
        time_horizon: int
            The time horizon window that
            controls Check Point occurence
            (unit: seconds),
        gen_plot: bool
            If True, triggers the plot
            generation,
        plot_path: str
            Path to save plots,
        gen_metrics: bool
            If True, triggers the
            metrics generation,
        metrics_path: str
            Path to save metrics,
        denstream_kwargs: dict
            Packed dictionary containing
            all DenStream parameters
        """
        self.gen_plot = gen_plot
        self.plot_path = plot_path
        self.gen_metrics = gen_metrics
        self.metrics_path = metrics_path
        self.event_count = 0
        self.total_cases = set()
        self.cases = []
        self.name = name
        self.time_horizon = time_horizon
        self.act_dic = {}
        self.initialized = False
        self.check_point = timestamp
        self.cp_count = 0
        self.nyquist = 0
        self.check_point_cases = 0
        self.process_model_graph = nx.DiGraph()
        self.denstream = DenStream(**denstream_kwargs)
        self.cluster_metrics = []
        self.case_metrics = []
        self.pmg_by_cp = []
        # don't think they will be useful, I haven't found them in any file of cdesf
        # self.initial_clusters = []
        # self.appeared_clusters = []

    def initialize_case_metrics(self) -> None:
        """
        Calculates GDtrace (graph_distance) and GDtime (time_distance)
        for cases in the first time horizon cycle.
        """
        for case in self.cases:
            graph_distance, time_distance = extract_case_distances(self.process_model_graph, case)
            case.graph_distance = graph_distance
            case.time_distance = time_distance

    def get_case(self, case_id: str) -> Union[int, None]:
        """
        Returns the case index from the cases list.

        Parameters
        --------------------------------------
        case_id: str
            The case identifier
        """
        for index, case in enumerate(self.cases):
            if case.id == case_id:
                return index
        return None

    def release_cases_from_memory(self) -> None:
        """
        Releases older cases based on the Nyquist value.
        The result is stored in self.cases.
        """
        self.cases = self.cases[:self.nyquist]

    def initialize_cdesf(self) -> None:
        """
        Initializes system variables, creates the first Process Model Graph
        and initializes DenStream.
        """
        self.nyquist = self.check_point_cases * 2
        # initialize PMG
        self.process_model_graph = initialize_graph(nx.DiGraph(), self.cases)
        # compute case metrics for initial cases
        self.initialize_case_metrics()

        # initialise denstream
        self.denstream.dbscan(self.cases)

        # plot
        # if self.gen_plot:
        #    self.gen_plots()

        # metrics
        # if self.gen_metrics:
        #     self.gen_cluster_metrics()
        #     self.save_pmg_on_check_point()
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
        Updates the checkpoint
        """
        self.cp_count += 1

        if len(self.cases) > self.nyquist:
            """
            Recalculates nyquist, releases cases and updates model (merges graphs)
            """
            self.release_cases_from_memory()
            self.nyquist = self.check_point_cases * 2

            check_point_graph = initialize_graph(nx.DiGraph(), self.cases)
            self.process_model_graph = merge_graphs(self.process_model_graph, check_point_graph)

        self.check_point_cases = 0

    def process_event(self, case_id: str, act_name: str, act_timestamp: datetime, event_index: int) -> None:
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
        event_index: int
            Index of current event
        """
        self.event_count += 1
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

            # visualization part
            # plots
            # if self.gen_plot:
            #    self.gen_plots()

            # metrics
            # if self.gen_metrics:
            #    self.gen_cluster_metrics()
            #    self.gen_case_metrics(event_index, index)

            if (current_time - self.check_point).total_seconds() > self.time_horizon:
                """
                Check point
                """
                self.check_point = current_time
                self.check_point_update()

                # metrics
                if self.gen_metrics:
                    self.save_pmg_on_check_point()

    def gen_case_metrics(self, event_index: int, index: int) -> None:
        """
        Generates case metrics and saves them in the
        self._case_metrics attribute.

        Parameters
        --------------------------------------
        event_index: int
            Index of the current event
        index: int
            Index of the current case
        """
        self.case_metrics.append([event_index,
                                  self.cases[index].id,
                                  self.cases[index].graph_distance,
                                  self.cases[index].time_distance])

    def save_pmg_on_check_point(self) -> None:
        """
        Saves the Process Model Graph at all check points in the
        self.pmg_by_cp attribute.
        TODO: improve how graphs are stored
        """
        if len(self.process_model_graph.edges) > 0:
            self.process_model_graph = normalize_graph(self.process_model_graph)
            for node1, node2, data in self.process_model_graph.edges(data=True):
                self.pmg_by_cp.append([self.cp_count, (node1, node2),
                                       data['weight'], data['time'],
                                       data['weight_normalized'], data['time_normalized']])

    # commented because part of visualization
    # def gen_plots(self):
    #    """
    #    Controls the plotting and handles all necessary arguments.
    #    """
    #    cases = self.get_case_points()
    #    cases_dict = {c.id: c for c in self.cases}
    #    points, outliers, c_clusters, p_clusters, o_clusters = \
    #        gen_data_plot(self.denstream, cases)
    #
    #    plot_clusters(process_name=self.name,
    #                  total_cases=len(self.total_cases),
    #                  cp=self.cp_count, outliers=outliers,
    #                  c_clusters=c_clusters,
    #                  p_clusters=p_clusters,
    #                  points=points,
    #                  th=self.time_horizon,
    #                  o_clusters=o_clusters,
    #                  n=f'{self.cp_count}_{self.event_count}',
    #                  epsilon=self.denstream.epsilon,
    #                  cases_dict=cases_dict,
    #                  event_index=self.event_count,
    #                  plot_path=self.plot_path)

    # I think it could be considered as visualization part
    #
    # def cluster_metrics(self) -> None:
    #    """
    #    Converts self._cluster_metrics into a dataframe and saves it.
    #    """
    #    df = pd.DataFrame(self.cluster_metrics,
    #                      columns=['index', 'centroid_x',
    #                               'centroid_y', 'radius',
    #                               'weight', 'total_cases', 'cp',
    #                               'clusterType', 'clusterID'])

    #    df.to_csv(f'{self.metrics_path}/{self.name}_cluster_metrics.csv',
    #              index=False)

    # def case_metrics(self) -> None:
    #    """
    #    Converts self._case_metrics into a dataframe and saves it.
    #    """
    #    df = pd.DataFrame(self.case_metrics, columns=['index', 'case_id',
    #                                                  'gdtrace', 'gdtime'])
    #    df.to_csv(f'{self.metrics_path}/{self.name}_case_metrics.csv',
    #              index=False)

    # def pmg_state_by_cp(self) -> None:
    #    """
    #    Converts self._pmg_by_cp into a dataframe and saves it.
    #    """
    #    df = pd.DataFrame(self.pmg_by_cp,
    #                      columns=['cp', 'transition', 'weight', 'time',
    #                               'weight_norm', 'time_norm'])
    #    df.to_csv(f'{self.metrics_path}/{self.name}_pmgs.csv', index=False)

    # no more needed, thought
    # def get_case_points(self):
    #    """
    #    Returns the point attribute of all cases in cases.
    #    """
    #    cases_points = [c.point for c in self.cases]
    #    return cases_points

    # commented because part of visualization
    # def gen_cluster_metrics(self):
    #    """
    #    Generates cluster metrics and saves them in the
    #    self._cluster_metrics attribute.
    #    """
    #    cases = self.get_case_points()
    #    points, outliers, c_clusters, p_clusters, o_clusters = \
    #        gen_data_plot(self.denstream, cases)
    #
    #    self.cluster_metrics.extend(cluster_metrics(
    #        total_cases=len(self.total_cases),
    #        event_index=self.event_count,
    #        cp=self.cp_count,
    #        c_clusters=c_clusters,
    #        p_clusters=p_clusters,
    #        o_clusters=o_clusters))
