import numpy as np
from datetime import datetime
from pm4py.objects.log.obj import EventStream
from pm4py.util import constants


class Case:
    """
    Represents a case and stores its attributes, such as activities,
    timestamps, graph_distance, time_distance, among others.
    """

    id: str
    events: EventStream
    distances: "dict[str, float]"

    def __init__(self, id: str):
        """
        Receives a case identifier and initializes the attributes of a new case.

        Parameters
        --------------------------------------
        id: str,
            Case identifier
        """
        self.id = id
        self.events = []
        self.distances = {}

    @property
    def point(self) -> np.ndarray:
        if len(self.distances) == 0:
            return np.zeros(2)

        return np.array(list(self.distances.values()))

    @property
    def last_time(self) -> datetime:
        """
        Returns
        --------------------------------------
        Retrieves the last event timestamp
        """
        return self.events[-1]["time:timestamp"]

    def add_event(self, event) -> None:
        # TODO: Try to extract correct type for an element of `EventStream`.
        # On TypeScript it would look something like `type MyType = EventStream[number]`.
        self.events.append(event)

    def get_trace(self) -> "list[str]":
        return [event["concept:name"] for event in self.events]

    def get_timestamps(self) -> "list[datetime]":
        return [event["time:timestamp"] for event in self.events]

    def get_attribute(self, attribute: str) -> "list":
        return [event.get(attribute) for event in self.events]

    # def set_activity(self, activity_name: str, activity_timestamp: datetime) -> None:
    #     """
    #     Creates a new Activity and appends it to the case activities list.

    #     Parameters
    #     --------------------------------------
    #     activity_name: str,
    #         Name of the activity
    #     activity_timestamp: datetime,
    #         Time of activity conclusion
    #     """
    #     activity = Activity(activity_name, activity_timestamp)
    #     self.activities.append(activity)

    # def get_trace(self) -> List[str]:
    #     """
    #     Returns
    #     --------------------------------------
    #     List of activity names
    #     """
    #     return [activity.name for activity in self.activities]

    # def get_timestamps(self) -> List[datetime.datetime]:
    #     """
    #     Returns
    #     --------------------------------------
    #     List of activity timestamps
    #     """
    #     return [activity.timestamp for activity in self.activities]
