from datetime import datetime
from cdesf2.utils import time_difference


def test_time_difference():
    timestamps_list = []
    timestamps_list.append([datetime(2015, 5, 10, 8, 30, 30),
                            datetime(2015, 5, 10, 8, 45, 30),
                            datetime(2015, 5, 10, 8, 50, 30),
                            datetime(2015, 5, 10, 8, 53, 30),
                            datetime(2015, 5, 10, 8, 59, 30),
                            datetime(2015, 5, 10, 9, 1, 30)])

    timestamps_list.append([datetime(2015, 5, 10, 9, 5, 30),
                            datetime(2015, 5, 10, 9, 55, 30),
                            datetime(2015, 5, 10, 9, 57, 30),
                            datetime(2015, 5, 10, 9, 59, 59)])

    timestamps_list.append([datetime(2015, 5, 10, 10, 5, 30),
                            datetime(2015, 5, 10, 11, 5, 30),
                            datetime(2015, 5, 10, 12, 5, 30),
                            datetime(2015, 5, 10, 14, 9, 59)])

    timestamps_list.append([datetime(2015, 5, 10, 10, 5, 30, 552),
                            datetime(2015, 6, 10, 11, 5, 30, 42),
                            datetime(2015, 8, 10, 12, 5, 30, 154),
                            datetime(2016, 5, 10, 14, 9, 59, 5)])

    true_values = [[900, 300, 180, 360, 120],
                   [3000, 120, 149],
                   [3600, 3600, 7469],
                   [2681999.99949, 5274000.000112, 23681068.999851]]

    results = time_difference(timestamps_list)

    assert len(results) == len(timestamps_list)
    assert results == true_values


def test_time_difference_2():
    timestamps_list = []

    results = time_difference(timestamps_list)
    assert results[0][0] == 0

    timestamps_list.append([datetime(2015, 5, 10, 8, 30, 30)])
    timestamps_list.append([datetime(2015, 5, 10, 8, 40, 30)])
    timestamps_list.append([datetime(2015, 5, 10, 8, 50, 30)])

    results = time_difference(timestamps_list)
    assert results == [[0], [0], [0]]
