import data_structures.Graph as Graph
import numpy as np
import pandas as pd
from data_structures import Case
from datetime import datetime as dt
from clustering.denstream import Case as CasePoint
from clustering.denstream import DenStream, gen_data_plot, plot_clusters, cluster_metrics

from collections import deque


class Process:
    """
    The Process class deals with the core concepts of CDESF controlling several
    operations such as the setting of a case, graphs management (Nyquist),
    DenStream triggering, plotting results and metrics recording.
    """
    def __init__(self, name, timestamp, th, gen_plot, plot_path,
                 gen_metrics, metrics_path, denstream_kwargs):
        """
        This function sets up a new process, defining its name,
        and preparing initial attributes.
        """
        self._gen_plot = gen_plot
        self._plot_path = plot_path
        self._gen_metrics = gen_metrics
        self._metrics_path = metrics_path
        self._event_count = 0
        self._total_cases = set()
        self._cases = []
        self._name = name
        self._th = th
        self._act_dic = {}
        self._possible_act_names = deque(['a', 'b', 'c', 'd', 'e', 'f',
                                          'g', 'h', 'i', 'j', 'k', 'l',
                                          'm', 'n', 'o', 'p', 'q', 'r',
                                          's', 't', 'u', 'v', 'w', 'x',
                                          'y', 'z'])
        self._gpCreation = False
        self._check_point = dt.strptime(timestamp, '%Y/%m/%d %H:%M:%S.%f')
        self._cp_count = 0
        self._nyquist = 0
        self._cp_cases = 0
        self._process_graph = {}
        self._denstream = DenStream(**denstream_kwargs)
        self._cluster_metrics = []
        self._case_metrics = []
        self._pmg_by_cp = []
        self._initial_clusters = []
        self._appeared_clusters = []

    def convertAct(self, act_name):
        """
        Receives an activity name and converts
        it using the "possible_act_names" deque.
        """
        if act_name not in self._act_dic:
            self._act_dic[act_name] = self._possible_act_names.popleft()
        return self._act_dic[act_name]

    def getListForGP(self, case_id):
        """
        Retrieves a list of cases from the first check point
        excluding the case passed as an argument.
        """
        trace_list = []
        timestamp_list = []
        for case in self._cases:
            if case._id != case_id:
                trace_list.append(case._trace)
                timestamp_list.append(case._timestamp)
        return trace_list, timestamp_list

    def getList(self):
        """
        Retrieves a list of cases traces and timestamps.
        """
        trace_list = []
        timestamp_list = []
        for case in self._cases:
            trace_list.append(case._trace)
            timestamp_list.append(case._timestamp)
        return trace_list, timestamp_list

    def getCase(self, case_id):
        """
        Returns the case index from the cases list.
        """
        for index, case in enumerate(self._cases):
            if case._id == case_id:
                return index
        return None

    def delCases(self):
        """
        Releases older cases based on the Nyquist value.
        The result is stored in self._cases.
        """
        cases_sorted = sorted(self._cases, key=lambda x: x.getLastTime(), reverse=True)
        self._cases = cases_sorted[:self._nyquist]

    def GPova(self):
        """
        Calculates GDtrace (gwd) and GDtime (twd) for cases in the first time
        horizon cycle.
        For every case, a graph is constructed with all other cases and the
        metrics computed.
        Note that none of this graphs are the Process Model Graph, instead they
        are only auxiliary graphs used once for metrics extraction for the
        initial cases.
        """
        for case in self._cases:
            traces, timestamps = self.getListForGP(case._id)
            gp_graph = Graph.createGraph(traces, timestamps)

            gwd, twd = Graph.computeFeatures(gp_graph, case._trace,
                                             case._timestamp)
            case.setGwd(gwd)
            case.setTwd(twd)

    def getCasePoints(self):
        """
        Returns the point attribute of all cases in cases.
        """
        cases_points = [c._point for c in self._cases]
        return cases_points

    def genClusterMetrics(self):
        """
        Generates cluster metrics and saves them in the
        self._cluster_metrics attribute.
        """
        cases = self.getCasePoints()
        points, outliers, c_clusters, p_clusters, o_clusters = \
            gen_data_plot(self._denstream, cases)

        self._cluster_metrics.extend(cluster_metrics(
                                    total_cases=len(self._total_cases),
                                    event_index=self._event_count,
                                    cp=self._cp_count,
                                    c_clusters=c_clusters,
                                    p_clusters=p_clusters,
                                    o_clusters=o_clusters))

    def genCaseMetrics(self, event_index, index):
        """
        Generates case metrics and saves them in the
        self._case_metrics attribute.
        """
        self._case_metrics.append([event_index,
                                   self._cases[index]._id,
                                   self._cases[index]._gwd,
                                   self._cases[index]._twd])

    def genPmgMetrics(self):
        """
        Saves the Process Model Graph at all check points in the
        self._pmg_by_cp attribute.
        """
        if len(self._process_graph) > 0:
            self._process_graph = Graph.normGraph(self._process_graph)
            for v in self._process_graph.values():
                self._pmg_by_cp.append([self._cp_count, v._name,
                                        v._weight, v._time,
                                        v._weight_norm, v._time_norm])

    def genPlots(self):
        """
        Controls the plotting and handles all necessary arguments.
        """
        cases = self.getCasePoints()
        cases_dict = {c._id: c for c in self._cases}
        points, outliers, c_clusters, p_clusters, o_clusters = \
            gen_data_plot(self._denstream, cases)

        plot_clusters(process_name=self._name,
                      total_cases=len(self._total_cases),
                      cp=self._cp_count, outliers=outliers,
                      c_clusters=c_clusters,
                      p_clusters=p_clusters,
                      points=points,
                      th=self._th,
                      o_clusters=o_clusters,
                      n=f'{self._cp_count}_{self._event_count}',
                      epsilon=self._denstream._epsilon,
                      cases_dict=cases_dict,
                      event_index=self._event_count,
                      plot_path=self._plot_path)

    def initialiseCDESF(self, current_time):
        """
        Initializes system variables, creates the first Process Model Graph
        and initializes DenStream.
        """
        self._nyquist = self._cp_cases * 2
        self._check_point = current_time
        self._gpCreation = True

        traces, timestamps = self.getList()
        self._process_graph = Graph.createGraph(traces, timestamps)
        self.GPova()

        cases = self.getCasePoints()

        # initialise denstream
        self._denstream.DBSCAN(cases)

        # plot
        if self._gen_plot:
            self.genPlots()

        # metrics
        if self._gen_metrics:
            self.genClusterMetrics()
            self.genPmgMetrics()

    def setCase(self, case_id, act_name, act_timestamp, event_index):
        """
        The core function in the Process class.
        Sets a new case and controls the check point.
        If gpCreation is True, calculates the case metrics and uses
        them to train DenStream, recalculates the Nyquist and releases
        old cases if necessary.
        """
        self._event_count += 1
        self._total_cases.add(case_id)

        # check if case exists and creates one if it doesn't
        index = self.getCase(case_id)
        if index is None:
            self._cases.append(Case(case_id))
            self._cp_cases += 1
            index = self.getCase(case_id)
        # add activity
        self._cases[index].setActivity(act_name, act_timestamp)
        # act_conv = act_name
        act_conv = self.convertAct(act_name)
        self._cases[index]._trace.append(act_conv)
        self._cases[index]._timestamp.append(act_timestamp)
        # reorder list, putting the newest case in the first position
        self._cases.append(self._cases.pop(index))

        current_time = dt.strptime(self._cases[index].getLastTime(),
                                   '%Y/%m/%d %H:%M:%S.%f')


        if ((current_time - self._check_point).total_seconds() > self._th and
            not self._gpCreation):
            """
            Checks the first check point for CDESF initialization
            """
            self.initialiseCDESF(current_time)
        elif self._gpCreation:
            """
            If we are past the first check point, graph distances are calculated
            and DenStream triggered
            """
            gwd, twd = Graph.computeFeatures(self._process_graph,
                                             self._cases[index]._trace,
                                             self._cases[index]._timestamp)

            self._cases[index].setGwd(gwd)
            self._cases[index].setTwd(twd)

            # DENSTREAM
            self._denstream.train(self._cases[index]._point)

            # plots
            if self._gen_plot:
                self.genPlots()

            # metrics
            if self._gen_metrics:
                self.genClusterMetrics()
                self.genCaseMetrics(event_index, index)

            if (current_time - self._check_point).total_seconds() > self._th:
                """
                Check point
                """
                self._check_point = current_time
                self._cp_count += 1

                if len(self._cases) > self._nyquist:
                    """
                    Recalculates nyquist, releases cases,
                    updates model (merges graphs)
                    """
                    self.delCases()
                    self._nyquist = self._cp_cases*2

                    tr, ti = self.getList()
                    cp_graph = Graph.createGraph(tr, ti)
                    Graph.mergeGraphs(self._process_graph, cp_graph)

                self._cp_cases = 0

                # metrics
                if self._gen_metrics:
                    self.genPmgMetrics()

    def clusterMetrics(self):
        """
        Converts self._cluster_metrics into a dataframe and saves it.
        """
        df = pd.DataFrame(self._cluster_metrics,
                          columns=['index', 'centroid_x',
                                   'centroid_y', 'radius',
                                   'weight', 'total_cases', 'cp',
                                   'clusterType', 'clusterID'])

        df.to_csv(f'{self._metrics_path}/{self._name}_cluster_metrics.csv',
                  index=False)

    def caseMetrics(self):
        """
        Converts self._case_metrics into a dataframe and saves it.
        """
        df = pd.DataFrame(self._case_metrics, columns=['index', 'case_id',
                                                       'gdtrace', 'gdtime'])
        df.to_csv(f'{self._metrics_path}/{self._name}_case_metrics.csv',
                  index=False)

    def pmgStateByCp(self):
        """
        Converts self._pmg_by_cp into a dataframe and saves it.
        """
        df = pd.DataFrame(self._pmg_by_cp,
                          columns=['cp', 'transition',  'weight', 'time',
                                   'weight_norm', 'time_norm'])
        df.to_csv(f'{self._metrics_path}/{self._name}_pmgs.csv', index=False)
