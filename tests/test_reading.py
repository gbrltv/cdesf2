import pandas as pd
import numpy as np
from datetime import datetime

from cdesf2.utils import (read_csv, read_csv_pm4py)


def test_read_csv():
    path = 'demo'
    filename = 'Detail_Supplier_IW-Frozen.csv'

    event_stream_test = read_csv(f'{path}/{filename}')

    assert isinstance(event_stream_test, np.ndarray)
    assert len(event_stream_test) == 5000

    first_event = event_stream_test[0]
    assert first_event[0] == 'Case 496'
    assert first_event[1] == 'Process_Creation'
    assert isinstance(first_event[2], datetime)
    assert first_event[2] == datetime(2010, 9, 21, 9, 0, 13)

    second_event = event_stream_test[1]
    assert second_event[0] == 'Case 12186'
    assert second_event[1] == 'Process_Creation'
    assert isinstance(second_event[2], datetime)
    assert second_event[2] == datetime(2010, 9, 21, 9, 0, 21)

    second_to_last_event = event_stream_test[4998]
    assert second_to_last_event[0] == 'Case 8848'
    assert second_to_last_event[1] == 'ME_Fabrication_Checker'
    assert isinstance(second_to_last_event[2], datetime)
    assert second_to_last_event[2] == datetime(2011, 4, 27, 8, 0, 59)

    last_event = event_stream_test[4999]
    assert last_event[0] == 'Case 10634'
    assert last_event[1] == 'ME_Fabrication_Checker'
    assert isinstance(last_event[2], datetime)
    assert last_event[2] == datetime(2011, 4, 27, 9, 0, 0)

    event_stream = pd.read_csv(f'{path}/{filename}',
                               usecols=['case', 'activity', 'timestamp'],
                               parse_dates=['timestamp'],
                               infer_datetime_format=True,
                               memory_map=True)
    event_stream['activity'].replace(' ', '_', regex=True, inplace=True)

    assert np.all(event_stream.values == event_stream_test)


def test_read_csv_pm4py():
    path = 'demo/Detail_Supplier_IW-Frozen.csv'
    trace = read_csv_pm4py(path)

    assert len(trace) == 1262

    events = []
    for case in trace:
        # TODO: Is `_list` the best way to access events in a `Trace`?
        for event in case._list:
            events.append(event)

    assert len(events) == 5000
