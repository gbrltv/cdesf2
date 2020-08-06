from datetime import datetime


class Activity:
    """
    The lowest level representation of an activity.
    """
    def __init__(self, name: str, timestamp: datetime):
        """
        Initializes an Activity setting its name and timestamp.

        Parameters
        --------------------------------------
        name: str,
            Name of the activity
        timestamp: datetime,
            Time of activity conclusion
        Returns
        --------------------------------------
        """
        self.name = name
        self.timestamp = timestamp
