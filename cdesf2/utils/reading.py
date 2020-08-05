import pandas as pd
import numpy as np


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

