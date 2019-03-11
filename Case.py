import numpy as np
from Activity import Activity
from denstream import Case as CasePoint

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
        point = np.array([self._gwd, self._twd])
        self._point = CasePoint(id=self._id, point=point)

    # def __repr__(self):
    #     return f'{self._id}, {self._timestamp[-1]}'

    def setActivity(self, act_name, act_timestamp):
        """
        Creates a new Activity and appends it to self._activities.
        """
        activity = Activity(act_name, act_timestamp)
        self._activities.append(activity)

    def getLastTime(self):
        """
        Retrieves the last event timestamp and is
        used to sort cases before being deleted.
        """
        return self._timestamp[-1]

    def setGwd(self, gwd):
        """
        Receives a value corresponding to GDtrace and stores
        it in both self._gwd and self._point attributes.
        """
        self._gwd = gwd
        self._point.point[0] = gwd

    def setTwd(self, twd):
        """
        Receives a value corresponding to GDtime and stores
        it in both self._twd and self._point attributes.
        """
        self._twd = twd
        self._point.point[1] = twd
