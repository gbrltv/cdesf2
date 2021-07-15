import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pandas._libs.tslibs import Timestamp
from pm4py.objects.log import obj

from cdesf.utils import read_csv, read_xes


def test_read_csv():
    # TODO: Is there a reason to replace " " in the activity name with "_"?
    event_stream_test = read_csv("demo/Detail_Supplier_IW-Frozen.csv")

    assert isinstance(event_stream_test, obj.EventStream)
    assert len(event_stream_test) == 5000

    first_event = event_stream_test[0]
    assert first_event["case:concept:name"] == "Case 496"
    assert first_event["concept:name"] == "Process Creation"
    assert isinstance(first_event["time:timestamp"], datetime)
    assert first_event["time:timestamp"] == datetime(2010, 9, 21, 9, 0, 13)

    second_event = event_stream_test[1]
    assert second_event["case:concept:name"] == "Case 12186"
    assert second_event["concept:name"] == "Process Creation"
    assert isinstance(second_event["time:timestamp"], datetime)
    assert second_event["time:timestamp"] == datetime(2010, 9, 21, 9, 0, 21)

    second_to_last_event = event_stream_test[-2]
    assert second_to_last_event["case:concept:name"] == "Case 8848"
    assert second_to_last_event["concept:name"] == "ME Fabrication Checker"
    assert isinstance(second_to_last_event["time:timestamp"], datetime)
    assert second_to_last_event["time:timestamp"] == datetime(2011, 4, 27, 8, 0, 59)

    last_event = event_stream_test[-1]
    assert last_event["case:concept:name"] == "Case 10634"
    assert last_event["concept:name"] == "ME Fabrication Checker"
    assert isinstance(last_event["time:timestamp"], datetime)
    assert last_event["time:timestamp"] == datetime(2011, 4, 27, 9, 0, 0)


def test_read_xes():
    event_stream_test = read_xes("demo/running-example.xes")

    assert isinstance(event_stream_test, obj.EventStream)
    assert len(event_stream_test) == 42

    tzinfo = timezone(timedelta(seconds=3600))

    first_event = event_stream_test[0]
    assert first_event["case:concept:name"] == "1"
    assert first_event["concept:name"] == "register request"
    assert isinstance(first_event["time:timestamp"], datetime)
    assert first_event["time:timestamp"] == datetime(2010, 12, 30, 11, 2, tzinfo=tzinfo)

    second_event = event_stream_test[1]
    assert second_event["case:concept:name"] == "1"
    assert second_event["concept:name"] == "examine thoroughly"
    assert isinstance(second_event["time:timestamp"], datetime)
    assert second_event["time:timestamp"] == datetime(
        2010, 12, 31, 10, 6, tzinfo=tzinfo
    )

    second_to_last_event = event_stream_test[-2]
    assert second_to_last_event["case:concept:name"] == "4"
    assert second_to_last_event["concept:name"] == "decide"
    assert isinstance(second_to_last_event["time:timestamp"], datetime)
    assert second_to_last_event["time:timestamp"] == datetime(
        2011, 1, 9, 12, 2, tzinfo=tzinfo
    )

    last_event = event_stream_test[-1]
    assert last_event["case:concept:name"] == "4"
    assert last_event["concept:name"] == "reject request"
    assert isinstance(last_event["time:timestamp"], datetime)
    assert last_event["time:timestamp"] == datetime(2011, 1, 12, 15, 44, tzinfo=tzinfo)
