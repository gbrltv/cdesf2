import pandas as pd
import numpy as np
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants
from pm4py.objects.log import log


def read_csv(path: str) -> np.ndarray:
    """
    Reads the .csv file, extracts the events and preprocesses data for CDESF

    Parameters
    --------------------------------------
    path: str
        File path and name
    Returns
    --------------------------------------
    An array of events
    """
    event_stream = pd.read_csv(path,
                               usecols=['case', 'activity', 'timestamp'],
                               parse_dates=['timestamp'],
                               infer_datetime_format=True,
                               memory_map=True)
    event_stream['activity'].replace(' ', '_', regex=True, inplace=True)
    return event_stream.values


def read_csv_pm4py(path: str) -> log.EventLog:
    log_csv = pd.read_csv(path, sep=',')
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    log_csv = log_csv.sort_values('timestamp')

    event_stream = log_converter.apply(log_csv, parameters={
        constants.PARAMETER_CONSTANT_CASEID_KEY: 'case',
        constants.PARAMETER_CONSTANT_ACTIVITY_KEY: 'activity',
        constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: 'timestamp',
    })

    return event_stream
