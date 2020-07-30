import pandas as pd
import numpy as np


def reading_csv(path: str, filename: str) -> np.ndarray:
    """
    This function reads the .csv file, extract
    the events and made pre-processing to make data
    suitable to CDESF

    Parameters
    --------------------------------------
    path: str
        Path where file is stored
    filename:str
        File to read

    Returns
    --------------------------------------
    An array of events
    """
    event_stream = pd.read_csv(f'{path}/{filename}',
                               usecols=['Case ID', 'Activity', 'Complete Timestamp'],
                               parse_dates=['Complete Timestamp'],
                               infer_datetime_format=True,
                               memory_map=True)
    event_stream['Activity'].replace(' ', '_', regex=True, inplace=True)
    return event_stream.values

