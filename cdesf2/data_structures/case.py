import numpy as np
import datetime as datetime
from typing import List
from .activity import Activity


class Case:
    """
    Represents a case and stores its attributes, such as activities,
    timestamps, graph_distance, time_distance, among others.
    """
    def __init__(self, case_id: str):
        """
        Receives a case identifier and initializes the attributes of a new case.

        Parameters
        --------------------------------------
        case_id: str,
            Case identifier
        """
        self.id = case_id
        self.activities = []
        self.graph_distance = np.nan
        self.time_distance = np.nan

    @property
    def point(self):
        return np.array([self.graph_distance, self.time_distance])

    @property
    def last_time(self) -> datetime:
        """
        Returns
        --------------------------------------
        Retrieves the last event timestamp
        """
        return self.activities[-1].timestamp

    def set_activity(self, activity_name: str, activity_timestamp: datetime) -> None:
        """
        Creates a new Activity and appends it to the case activities list.

        Parameters
        --------------------------------------
        activity_name: str,
            Name of the activity
        activity_timestamp: datetime,
            Time of activity conclusion
        """
        activity = Activity(activity_name, activity_timestamp)
        self.activities.append(activity)

    def get_trace(self) -> List[str]:
        """
        Returns
        --------------------------------------
        List of activity names
        """
        return [activity.name for activity in self.activities]

    def get_timestamps(self) -> List[datetime.datetime]:
        """
        Returns
        --------------------------------------
        List of activity timestamps
        """
        return [activity.timestamp for activity in self.activities]
