from cdesf2.data_structures.activity import Activity
import pytest


class TestActivity:
    def test_initial_value(self):
        activity = Activity('activity1', 'timestamp1')
        assert type(activity.name) is str
        assert type(activity.timestamp) is str
        assert activity.name == 'activity1'
        assert activity.timestamp == 'timestamp1'

    def test_no_value(self):
        with pytest.raises(Exception) as e_info:
            empty_activity = Activity()
