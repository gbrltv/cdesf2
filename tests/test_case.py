import numpy as np
from datetime import datetime
from cdesf2.data_structures.case import Case
from cdesf2.data_structures.activity import Activity
import pytest


def test_case():
    case = Case('1')
    assert type(case) is Case


def test_initial_value():
    case = Case('1')
    assert type(case.id) is str
    assert case.id == '1'
    assert type(case.activities) is list
    assert case.activities == []
    assert case.graph_distance is np.nan
    assert case.time_distance is np.nan
    # assert case.point[0] == '1'
    # assert np.all(case.point[1] == [0, 0])


def test_no_value():
    with pytest.raises(Exception) as e_info:
        empty_case = Case()


def test_set_activity():
    activity = Activity('activity1', datetime.strptime('2015/05/10 08:22:53.000', '%Y/%m/%d %H:%M:%S.%f'))
    case = Case('2')
    case.set_activity(activity.name, activity.timestamp)

    assert type(case.activities[0]) is Activity
    assert case.activities[0].name == 'activity1'
    assert case.activities[0].timestamp == datetime(2015, 5, 10, 8, 22, 53)


def test_get_last_time():
    case = Case('3')

    case.set_activity('activity1', datetime(2015, 5, 10, 8, 30, 00))
    case.set_activity('activity2', datetime(2015, 5, 10, 9, 00, 00))
    case.set_activity('activity3', datetime(2015, 5, 10, 9, 30, 00))
    case.set_activity('activity4', datetime(2015, 5, 10, 10, 00, 00))

    assert case.get_last_time() == datetime(2015, 5, 10, 10, 00, 00)


def test_get_trace():
    case = Case('4')
    activity_names = ['activity1', 'activity2', 'activity3', 'activity4']

    case.set_activity(activity_names[0], datetime(2015, 5, 10, 8, 30, 00))
    case.set_activity(activity_names[1], datetime(2015, 5, 10, 9, 00, 00))
    case.set_activity(activity_names[2], datetime(2015, 5, 10, 9, 30, 00))
    case.set_activity(activity_names[3], datetime(2015, 5, 10, 10, 00, 00))

    assert case.get_trace() == activity_names


def test_get_timestamps():
    case = Case('5')
    timestamps = [datetime(2015, 5, 10, 8, 30, 00),
                  datetime(2015, 5, 10, 9, 00, 00),
                  datetime(2015, 5, 10, 9, 30, 00),
                  datetime(2015, 5, 10, 10, 00, 00)]

    case.set_activity('activity1', datetime(2015, 5, 10, 8, 30, 00))
    case.set_activity('activity2', datetime(2015, 5, 10, 9, 00, 00))
    case.set_activity('activity3', datetime(2015, 5, 10, 9, 30, 00))
    case.set_activity('activity4', datetime(2015, 5, 10, 10, 00, 00))

    assert case.get_timestamps() == timestamps


def test_distances():
    case = Case('6')

    assert case.graph_distance is np.nan
    assert case.time_distance is np.nan

    case.graph_distance = 6.57
    case.time_distance = 88

