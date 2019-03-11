import os
import pandas as pd
from CDESF import CDESF

"""
The main class sets the initial parameters of the framework which will be
used in subsequent classes. If the user intends to generate plots and metrics,
the main class sets up the directories and paths.



Notes:
The event log may be in CSV format. We use pandas library to read the event
log in a dataframe fashion. The columns must be case ID, activity name (event),
timestamp, process name. Event logs that do not conform to this will throw
an error and won't be consumed by the framework.
"""

path = 'demo'
process = 'Detail_Supplier_IW-Frozen.csv'
th = 259200
n_features = 2
lambda_ = 0.1
beta = 0.3
epsilon = 0.1
mu = 4
stream_speed = 1000
gen_plot = False
gen_metrics = True
plot_path = ''
metrics_path = ''

event_stream = pd.read_csv(f'{path}/{process}')
processes_names = set(event_stream['procedure_name'])
event_stream = event_stream.values

if gen_plot:
    try:
        for name in processes_names:
            plot_path = (f'plot/{name}/th_{th}/epsilon_{epsilon}/'
                         f'lambda_{lambda_}/beta_{beta}/mu_{mu}/stream_speed_{stream_speed}')
            os.makedirs('aux/dotfiles', exist_ok=True)
            os.makedirs('aux/graphvizfiles', exist_ok=True)
            os.makedirs(plot_path, exist_ok=True)
    except Exception as e:
        print(e)

if gen_metrics:
    try:
        for name in processes_names:
            metrics_path = (f'metrics/th_{th}/epsilon_{epsilon}/'
                            f'lambda_{lambda_}/beta_{beta}/'
                            f'mu_{mu}/stream_speed_{stream_speed}')
            os.makedirs(metrics_path, exist_ok=True)
    except Exception as e:
        print(e)

cdesf = CDESF(event_stream, th, gen_plot, plot_path,
              gen_metrics, metrics_path,
              denstream_kwargs={'n_features': n_features,
                                'beta': beta,
                                'lambda_': lambda_,
                                'epsilon': epsilon,
                                'mu': mu,
                            	'stream_speed': stream_speed})
cdesf.eventProcessing()
