from ..utils import extract_case_distances, extract_cases_time_and_trace,\
                    initialize_graph, normalize_graph, merge_graphs
import networkx as nx
import pandas as pd
from ..data_structures import Case
from ..clustering import DenStream
from typing import List, Union
from datetime import datetime



# from visualization import gen_data_plot, plot_clusters, cluster_metrics


class Process:
    """
    The Process class deals with the core concepts of CDESF controlling several
    operations such as the setting of a case, graphs management (Nyquist),
    DenStream triggering, plotting results and metrics recording.
    """

    def __init__(self, name: str, timestamp: datetime, th: int, gen_plot: bool, plot_path: str,
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
        th: int
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
        self.th = th
        self.act_dic = {}
        self.gp_creation = False
        self.check_point = timestamp
        self.cp_count = 0
        self.nyquist = 0
        self.cp_cases = 0
        self.process_model_graph = nx.DiGraph()
        self.denstream = DenStream(**denstream_kwargs)
        self.cluster_metrics = []
        self.case_metrics = []
        self.pmg_by_cp = []
        # don't think they will be useful, I haven't found them in any file of cdesf
        # self.initial_clusters = []
        # self.appeared_clusters = []

    def get_list_for_gp(self, case_id: str) -> List:
        """
        Retrieves a list of cases from the first check point
        excluding the case passed as an argument.

        Parameters
        --------------------------------------
        case_id: str
            The case identifier,
        Returns
        --------------------------------------
        case_list_before_cp: List
            The case list before check point
            without the case passed,
        """
        case_list_before_cp = []
        for case in self.cases:
            if case.id != case_id:
                case_list_before_cp.append(case)
        return case_list_before_cp

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

    def del_cases(self) -> None:
        """
        Releases older cases based on the Nyquist value.
        The result is stored in self._cases.
        """
        cases_sorted = sorted(self.cases, key=lambda x: x.get_last_time(), reverse=True)
        self.cases = cases_sorted[:self.nyquist]

    def g_pova(self) -> None:
        """
        Calculates GDtrace (graph_distance) and GDtime (time_distance)
        for cases in the first time horizon cycle.
        For every case, a graph is constructed with all other cases and the
        metrics computed.
        Note that none of this graphs are the Process Model Graph, instead they
        are only auxiliary graphs used once for metrics extraction for the
        initial cases.
        """
        for case in self.cases:
            gp_graph = initialize_graph(nx.DiGraph(), self.get_list_for_gp(case.id))

            graph_distance, time_distance = extract_case_distances(gp_graph, case)
            case.graph_distance = graph_distance
            case.time_distance = time_distance

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

    def gen_pmg_metrics(self) -> None:
        """
        Saves the Process Model Graph at all check points in the
        self._pmg_by_cp attribute.
        """
        if len(self.process_model_graph.edges) > 0:
            self.process_model_graph = normalize_graph(self.process_model_graph)
            for n1, n2, data in self.process_model_graph.edges(data=True):
                self.pmg_by_cp.append([self.cp_count, (n1, n2),
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
    #                  th=self.th,
    #                  o_clusters=o_clusters,
    #                  n=f'{self.cp_count}_{self.event_count}',
    #                  epsilon=self.denstream.epsilon,
    #                  cases_dict=cases_dict,
    #                  event_index=self.event_count,
    #                  plot_path=self.plot_path)

    def initialize_cdesf(self, current_time: datetime) -> None:
        """
        Initializes system variables, creates the first Process Model Graph
        and initializes DenStream.

        Parameters
        --------------------------------------
        current_time: datetime
            The last event time,
            used for check point making
        """
        self.nyquist = self.cp_cases * 2
        self.check_point = current_time
        self.gp_creation = True

        self.process_model_graph = initialize_graph(nx.DiGraph(), self.cases)
        self.g_pova()

        cases = self.cases

        # initialise denstream
        self.denstream.dbscan(cases)

        # plot
        # if self.gen_plot:
        #    self.gen_plots()

        # metrics
        if self.gen_metrics:
        #   self.gen_cluster_metrics()
            self.gen_pmg_metrics()

    def set_case(self, case_id: str, act_name: str,
                 act_timestamp: datetime, event_index: int) -> None:
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

        # check if case exists and creates one if it doesn't
        index = self.get_case(case_id)
        if index is None:
            self.cases.append(Case(case_id))
            self.cp_cases += 1
            index = self.get_case(case_id)
        # add activity
        self.cases[index].set_activity(act_name, act_timestamp)
        # reorder list, putting the newest case in the first position
        self.cases.append(self.cases.pop(index))

        current_time = self.cases[index].get_last_time()

        if ((current_time - self.check_point).total_seconds() > self.th and
                not self.gp_creation):
            """
            Checks the first check point for CDESF initialization
            """
            self.initialize_cdesf(current_time)
        elif self.gp_creation:
            """
            If we are past the first check point, graph distances are calculated
            and DenStream triggered
            """
            graph_distance, time_distance = extract_case_distances(self.process_model_graph, self.cases[index])
            self.cases[index].graph_distance = graph_distance
            self.cases[index].time_distance = time_distance

            # DENSTREAM
            self.denstream.train(self.cases[index])

            # visualization part
            # plots
            # if self.gen_plot:
            #    self.gen_plots()

            # metrics
            # if self.gen_metrics:
            #    self.gen_cluster_metrics()
            #    self.gen_case_metrics(event_index, index)

            if (current_time - self.check_point).total_seconds() > self.th:
                """
                Check point
                """
                self.check_point = current_time
                self.cp_count += 1

                if len(self.cases) > self.nyquist:
                    """
                    Recalculates nyquist, releases cases,
                    updates model (merges graphs)
                    """
                    self.del_cases()
                    self.nyquist = self.cp_cases * 2

                    check_point_graph = initialize_graph(nx.DiGraph(), self.cases)
                    merge_graphs(self.process_model_graph, check_point_graph)

                self.cp_cases = 0

                # metrics
                if self.gen_metrics:
                    self.gen_pmg_metrics()

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

    #def pmg_state_by_cp(self) -> None:
    #    """
    #    Converts self._pmg_by_cp into a dataframe and saves it.
    #    """
    #    df = pd.DataFrame(self.pmg_by_cp,
    #                      columns=['cp', 'transition', 'weight', 'time',
    #                               'weight_norm', 'time_norm'])
    #    df.to_csv(f'{self.metrics_path}/{self.name}_pmgs.csv', index=False)
