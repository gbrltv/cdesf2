import numpy as np
from datetime import datetime
from cdesf.data_structures.case import Case

import pytest


def test_case():
    case = Case("1")
    assert isinstance(case, Case)


def test_initial_value():
    case = Case("1")
    assert case.id == "1"
    assert len(case.events) == 0
    assert len(case.distances) == 0


# This test is replaced by the static analysis of the Python compiler.
# def test_no_value():
#     with pytest.raises(Exception):
#         assert Case()


def test_add_event():
    stream = [
        {
            "concept:name": "Activity A",
            "time:timestamp": datetime(2021, 4, 30, 9, 0, 0),
        }
    ]

    case = Case("test_add_event")
    case.add_event(stream[0])

    assert len(case.events) == 1
    assert case.events[0]["concept:name"] == "Activity A"
    assert case.events[0]["time:timestamp"] == datetime(2021, 4, 30, 9, 0, 0)


# def OLD_test_set_activity():
#     activity = Activity(
#         "activity1",
#         datetime.strptime("2015/05/10 08:22:53.000", "%Y/%m/%d %H:%M:%S.%f"),
#     )
#     case = Case("2")
#     case.set_activity(activity.name, activity.timestamp)

#     assert isinstance(case.activities[0], Activity)
#     assert case.activities[0].name == "activity1"
#     assert case.activities[0].timestamp == datetime(2015, 5, 10, 8, 22, 53)


def test_last_time():
    stream = [
        {
            "concept:name": "Activity A",
            "time:timestamp": datetime(2021, 4, 30, 9, 0, 0),
        },
        {
            "concept:name": "Activity B",
            "time:timestamp": datetime(2021, 4, 30, 10, 0, 0),
        },
        {
            "concept:name": "Activity C",
            "time:timestamp": datetime(2021, 5, 1, 9, 0, 0),
        },
    ]

    case = Case("test_last_time")
    case.add_event(stream[0])
    case.add_event(stream[1])
    case.add_event(stream[2])

    assert case.last_time == datetime(2021, 5, 1, 9, 0, 0)


# def OLD_test_last_time():
#     case = Case("3")

#     case.set_activity("activity1", datetime(2015, 5, 10, 8, 30, 00))
#     case.set_activity("activity2", datetime(2015, 5, 10, 9, 00, 00))
#     case.set_activity("activity3", datetime(2015, 5, 10, 9, 30, 00))
#     case.set_activity("activity4", datetime(2015, 5, 10, 10, 00, 00))

#     assert case.last_time == datetime(2015, 5, 10, 10, 00, 00)


def test_get_trace():
    stream = [
        {
            "concept:name": "Activity A",
            "time:timestamp": datetime(2021, 4, 30, 9, 0, 0),
        },
        {
            "concept:name": "Activity B",
            "time:timestamp": datetime(2021, 4, 30, 10, 0, 0),
        },
        {
            "concept:name": "Activity C",
            "time:timestamp": datetime(2021, 5, 1, 9, 0, 0),
        },
    ]

    case = Case("test_get_trace")
    case.add_event(stream[0])
    case.add_event(stream[1])
    case.add_event(stream[2])

    assert case.get_trace() == ["Activity A", "Activity B", "Activity C"]


# def OLD_test_get_trace():
#     case = Case("4")
#     activity_names = ["activity1", "activity2", "activity3", "activity4"]

#     case.set_activity(activity_names[0], datetime(2015, 5, 10, 8, 30, 00))
#     case.set_activity(activity_names[1], datetime(2015, 5, 10, 9, 00, 00))
#     case.set_activity(activity_names[2], datetime(2015, 5, 10, 9, 30, 00))
#     case.set_activity(activity_names[3], datetime(2015, 5, 10, 10, 00, 00))

#     assert case.get_trace() == activity_names


def test_get_timestamps():
    stream = [
        {
            "concept:name": "Activity A",
            "time:timestamp": datetime(2021, 4, 30, 9, 0, 0),
        },
        {
            "concept:name": "Activity B",
            "time:timestamp": datetime(2021, 4, 30, 10, 0, 0),
        },
        {
            "concept:name": "Activity C",
            "time:timestamp": datetime(2021, 5, 1, 9, 0, 0),
        },
    ]

    case = Case("test_get_timestamps")
    case.add_event(stream[0])
    case.add_event(stream[1])
    case.add_event(stream[2])

    assert case.get_timestamps() == [
        datetime(2021, 4, 30, 9, 00, 00),
        datetime(2021, 4, 30, 10, 00, 00),
        datetime(2021, 5, 1, 9, 00, 00),
    ]


# def OLD_test_get_timestamps():
#     case = Case("5")
#     timestamps = [
#         datetime(2015, 5, 10, 8, 30, 00),
#         datetime(2015, 5, 10, 9, 00, 00),
#         datetime(2015, 5, 10, 9, 30, 00),
#         datetime(2015, 5, 10, 10, 00, 00),
#     ]

#     case.set_activity("activity1", datetime(2015, 5, 10, 8, 30, 00))
#     case.set_activity("activity2", datetime(2015, 5, 10, 9, 00, 00))
#     case.set_activity("activity3", datetime(2015, 5, 10, 9, 30, 00))
#     case.set_activity("activity4", datetime(2015, 5, 10, 10, 00, 00))

#     assert case.get_timestamps() == timestamps


# def test_distances():
#     case = Case("6")

#     assert case.graph_distance is np.nan
#     assert case.time_distance is np.nan

#     case.graph_distance = 6.57
#     case.time_distance = 88
