import pandas as pd
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util import constants
from pm4py.objects.log import obj
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.importer.xes import importer as xes_importer


def read_csv(path: str) -> obj.EventStream:
    """
    Reads the .csv file, extracts the events and preprocesses data for CDESF.

    Parameters
    --------------------------------------
    path: str
        File path and name

    Returns
    --------------------------------------
    An event stream
    """

    log_csv = pd.read_csv(path, sep=",")

    # Mergesort is the only stable sorting algorithm.
    log_csv = log_csv.sort_values("timestamp", kind="mergesort")

    # Manually transform timestamps to datetimes.
    log_csv["timestamp"] = pd.to_datetime(log_csv["timestamp"])

    # Standardize column names across CSV and XES files.
    log_csv.rename(
        columns={
            "case": "case:concept:name",
            "activity": "concept:name",
            "timestamp": "time:timestamp",
        },
        inplace=True,
    )

    event_stream = log_converter.apply(
        log_csv, variant=log_converter.Variants.TO_EVENT_STREAM
    )

    return event_stream


def read_xes(path: str) -> obj.EventStream:
    """
    Reads the .xes file, extracts the events and preprocesses data for CDESF.

    Parameters
    --------------------------------------
    path: str
        File path and name

    Returns
    --------------------------------------
    An event stream
    """

    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    log_xes = xes_importer.apply(path, variant=variant, parameters=parameters)

    event_stream = log_converter.apply(
        log_xes,
        variant=log_converter.Variants.TO_EVENT_STREAM,
        parameters={
            constants.PARAMETER_CONSTANT_CASEID_KEY: "case:concept:name",
            constants.PARAMETER_CONSTANT_ACTIVITY_KEY: "concept:name",
            constants.PARAMETER_CONSTANT_TIMESTAMP_KEY: "time:timestamp",
        },
    )

    return event_stream
