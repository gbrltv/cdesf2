class Activity:
    """
    The lowest level representation of an activity.
    """
    def __init__(self, name, timestamp):
        """
        Initializes an Activity setting its name and timestamp.
        """
        self.name = name
        self.timestamp = timestamp
