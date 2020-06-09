import os
import re
import time
import psutil
import pandas as pd
from core.CDESF import CDESF

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

# hyperparameters configuration
path = '../demo'
process = 'Detail_Supplier_IW-Frozen.csv'
name = process.split('.')[0]
th = 43200
n_features = 2
lambda_ = 0.15
beta = 0.3
epsilon = 0.1
mu = 4
stream_speed = 1000
gen_plot = True
gen_metrics = True
plot_path = f'plot/{name}/th_{th}/epsilon_{epsilon}/lambda_{lambda_}/beta_{beta}/mu_{mu}/stream_speed_{stream_speed}'
metrics_path = f'metrics/{name}/th_{th}/epsilon_{epsilon}/lambda_{lambda_}/beta_{beta}/mu_{mu}/stream_speed_{stream_speed}'

start_time = time.time()
event_stream = pd.read_csv(f'{path}/{process}')
event_stream = event_stream.values

# folder for plotting
if gen_plot:
    try:
        os.makedirs('aux/dotfiles', exist_ok=True)
        os.makedirs('aux/graphvizfiles', exist_ok=True)
        os.makedirs(plot_path, exist_ok=True)
    except Exception as e:
        print(e)

# folder for metrics
if gen_metrics:
    try:
        os.makedirs(metrics_path, exist_ok=True)
    except Exception as e:
        print(e)

# cdesf instantiation
cdesf = CDESF(event_stream, th, gen_plot, plot_path,
              gen_metrics, metrics_path, name,
              denstream_kwargs={'n_features': n_features,
                                'beta': beta,
                                'lambda_': lambda_,
                                'epsilon': epsilon,
                                'mu': mu,
                                'stream_speed': stream_speed,
                                'ncluster': 0})
# cdesf processing
cdesf.eventProcessing()

elapsed_time = time.time() - start_time
mem = float(psutil.Process(os.getpid()).memory_full_info().uss)/1048576
print(f'Elapsed Time: {elapsed_time} seconds')
print(f'Memory Used: {mem} MBs')
