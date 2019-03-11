from Process import Process

class CDESF:
    """
    CDESF simulates the stream and handles the event to event management.
    If the event log contains more than one process, it can be handled properly.
    Moreover, this class calls the methods for metrics retrieval.
    """
    def __init__(self, stream, th, gen_plot, plot_path,
                 gen_metrics, metrics_path, denstream_kwargs):
        """
        Initializes the attributes.
        """
        self._stream = stream
        self._th = th
        self._gen_plot = gen_plot
        self._plot_path = plot_path
        self._gen_metrics = gen_metrics
        self._metrics_path = metrics_path
        self._denstream_kwargs = denstream_kwargs
        self._processes = {}

    def setProcess(self, name, case_id, act_name, act_timestamp, event_index):
        """
        This function receives the basic case attributes and sets them up
        accordingly.
        If a process is new in the stream, the function instantiates a new
        process object and adds it to the process dictionary.
        """
        if name not in self._processes:
            process = Process(name, act_timestamp, self._th, self._gen_plot,
                              self._plot_path, self._gen_metrics,
                              self._metrics_path, self._denstream_kwargs)
            self._processes[name] = process

        self._processes[name].setCase(case_id, act_name,
                                      act_timestamp, event_index)

    def eventProcessing(self):
        """
        Simulates the event stream by iterating through the stream variable,
        calls setProcess at each event and also controls the metrics recording.
        """
        for index, event in enumerate(self._stream):
            name, case_id, act_name, act_timestamp = (event[3], event[0],
                                                      event[1], event[2])
            self.setProcess(name, case_id, act_name, act_timestamp, index)
            
        if self._gen_metrics:
            for proc in self._processes:
                self._processes[proc].clusterMetrics()
                self._processes[proc].caseMetrics()
                self._processes[proc].pmgStateByCp()
