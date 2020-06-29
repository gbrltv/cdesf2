import numpy as np
from collections import namedtuple
from .activity import Activity


class Case:
    """
    Represents a case and stores its attributes, such as activities,
    timestamps, GDtrace (gwd), GDtime (twd), among others.
    """
    def __init__(self, case_id):
        """
        Receives a case identifier and initializes the attributes of a new case.
        """
        self._id = case_id
        self._activities = []
        self._trace = []
        self._timestamp = []
        self._gwd = 0
        self._twd = 0
        point_structure = np.array([self._gwd, self._twd])
        case_point = namedtuple('Case', ['id', 'point'])
        self._point = case_point(id=self._id, point=point_structure)

    def set_activity(self, act_name, act_timestamp):
        """
        Creates a new Activity and appends it to self._activities.
        """
        activity = Activity(act_name, act_timestamp)
        self._activities.append(activity)

    def get_last_time(self):
        """
        Retrieves the last event timestamp and is
        used to sort cases before being deleted.
        """
        return self._timestamp[-1]

    def set_gwd(self, gwd):
        """
        Receives a value corresponding to GDtrace and stores
        it in both self._gwd and self._point attributes.
        """
        self._gwd = gwd
        self._point.point[0] = gwd

    def set_twd(self, twd):
        """
        Receives a value corresponding to GDtime and stores
        it in both self._twd and self._point attributes.
        """
        self._twd = twd
        self._point.point[1] = twd
