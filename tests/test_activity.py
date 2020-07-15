from cdesf2.data_structures.activity import Activity
from datetime import datetime
import pytest


def test_initial_value():
    activity = Activity('activity1', datetime.strptime('2015/05/10 08:22:53.000', '%Y/%m/%d %H:%M:%S.%f'))
    assert type(activity.name) is str
    assert type(activity.timestamp) is datetime
    assert type(activity) is Activity
    assert activity.name == 'activity1'
    assert activity.timestamp == datetime(2015, 5, 10, 8, 22, 53)
    activity = Activity('activity 1', datetime.strptime('2015/05/10 08:22:53.000', '%Y/%m/%d %H:%M:%S.%f'))
    assert activity.name == 'activity_1'


def test_no_value():
    with pytest.raises(Exception):
        assert Activity()
