from typing import List, Tuple
from datetime import datetime
from ..data_structures import Case


def extract_cases_time_and_trace(case_list: List[Case]) -> Tuple[List[List[str]], List[List[datetime]]]:
    """
    Receives a list of cases and returns a list of traces and a list of timestamps

    Parameters
    --------------------------------------
    case_list: List[Case],
        List of cases
    Returns
    --------------------------------------
    trace_list: List[List[str]],
        List of traces
    time_list: List[List[datetime]],
        List of timestamps
    """
    if len(case_list) == 0:
        raise Exception("Extracting trace and timestamp list out of a list with no cases")

    trace_list, time_list = [], []
    for case in case_list:
        trace_list.append(case.get_trace())
        time_list.append(case.get_timestamps())

    return trace_list, time_list
