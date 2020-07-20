from datetime import datetime
from cdesf2.data_structures import Case
from cdesf2.utils import extract_cases_time_and_trace
import pytest


def test_extract_cases_time_and_trace():
    case_list = []

    with pytest.raises(Exception):
        assert extract_cases_time_and_trace(case_list)

    case = Case('1')
    case.set_activity('activity1', datetime(2015, 5, 10, 8, 30, 00))
    case.set_activity('activity2', datetime(2015, 5, 10, 9, 00, 00))
    case.set_activity('activity3', datetime(2015, 5, 10, 9, 30, 00))
    case.set_activity('activity4', datetime(2015, 5, 10, 10, 00, 00))
    case_list.append(case)

    case = Case('2')
    case.set_activity('activity1', datetime(2015, 5, 10, 8, 5, 30))
    case.set_activity('activity3', datetime(2015, 5, 10, 9, 33, 52))
    case.set_activity('activity2', datetime(2015, 5, 10, 10, 30, 00))
    case.set_activity('activity4', datetime(2015, 5, 10, 11, 50, 00))
    case_list.append(case)

    case = Case('3')
    case.set_activity('activity5', datetime(2016, 5, 10, 8, 5, 30))
    case.set_activity('activity1', datetime(2017, 5, 10, 9, 33, 52))
    case.set_activity('activity3', datetime(2018, 5, 10, 10, 30, 00))
    case.set_activity('activity4', datetime(2019, 5, 10, 11, 50, 00))
    case_list.append(case)

    trace_list, time_list = extract_cases_time_and_trace(case_list)

    expected_trace_list = [['activity1', 'activity2', 'activity3', 'activity4'],
                           ['activity1', 'activity3', 'activity2', 'activity4'],
                           ['activity5', 'activity1', 'activity3', 'activity4']]
    expected_time_list = [[datetime(2015, 5, 10, 8, 30, 00),
                           datetime(2015, 5, 10, 9, 00, 00),
                           datetime(2015, 5, 10, 9, 30, 00),
                           datetime(2015, 5, 10, 10, 00, 00)],
                          [datetime(2015, 5, 10, 8, 5, 30),
                           datetime(2015, 5, 10, 9, 33, 52),
                           datetime(2015, 5, 10, 10, 30, 00),
                           datetime(2015, 5, 10, 11, 50, 00)],
                          [datetime(2016, 5, 10, 8, 5, 30),
                           datetime(2017, 5, 10, 9, 33, 52),
                           datetime(2018, 5, 10, 10, 30, 00),
                           datetime(2019, 5, 10, 11, 50, 00)]]

    assert len(trace_list) == 3
    assert len(time_list) == 3
    assert trace_list == expected_trace_list
    assert time_list == expected_time_list
