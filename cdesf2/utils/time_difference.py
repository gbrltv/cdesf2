from datetime import datetime
from typing import List


def case_time_difference(case_timestamps: List[datetime]) -> List[float]:
    """
    Converts a list of case timestamps to timedeltas

    Parameters
    --------------------------------------
    case_timestamps: List[datetime],
        List of case activities datetimes
    Returns
    --------------------------------------
    time_delta: List[float],
        List of float time difference in seconds
    """
    if len(case_timestamps) <= 1:
        return [0]

    time_delta = []
    for i in range(len(case_timestamps)-1):
        time_delta.append((case_timestamps[i+1]-case_timestamps[i]).total_seconds())

    return time_delta


def time_difference(cases_timestamps: List[List[datetime]]) -> List[List[float]]:
    """
    Converts a list of a list of timestamps to a list of a list of timedeltas (in seconds)

    Parameters
    --------------------------------------
    cases_timestamps: List[List[datetime]],
        All cases timestamps
    Returns
    --------------------------------------
    case_time_difference: List[List[float]],
        All cases timedeltas
    """
    if len(cases_timestamps) == 0:
        return [[0]]

    cases_time_difference = []
    for case_timestamp in cases_timestamps:
        cases_time_difference.append(case_time_difference(case_timestamp))

    return cases_time_difference
