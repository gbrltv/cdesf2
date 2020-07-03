from cdesf2.data_structures.case import Case
import numpy as np
import pytest


class TestCase:
    @pytest.fixture
    def case(self):
        case = Case('1')
        return case

    def test_initial_value(self, case):
        assert type(case.id) is str
        assert case.id == '1'
        assert type(case.activities) is list
        assert type(case.trace) is list
        assert type(case.timestamp) is list
        assert case.activities == []
        assert case.trace == []
        assert case.timestamp == []
        assert case.gwd == 0
        assert case.twd == 0
        assert case.point[0] == '1'
        assert np.all(case.point[1] == [0, 0])

    def test_no_value(self):
        with pytest.raises(Exception) as e_info:
            empty_case = Case()

    def test_set_activity(self, case):
        case.set_activity('activity1', 'timestamp1')
        assert case.activities[0] == 'activity1'
        assert case.activities[1] == 'timestamp1'

    def test_get_last_time(self, case):
        case.timestamp.append('timestamp0')
        case.timestamp.append('timestamp1')
        case.timestamp.append('timestamp2')
        assert case.get_last_time() == 'timestamp2'

    def test_set_gwd(self, case):
        assert case.gwd == 0
        gwd = 2.5
        case.set_gwd(gwd)
        assert case.gwd == 2.5
        assert case.point.point[0] == 2.5

    def test_set_twd(self, case):
        assert case.twd == 0
        twd = 3.5
        case.set_twd(twd)
        assert case.twd == 3.5
        assert case.point.point[1] == 3.5
