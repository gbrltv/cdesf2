class Transition:
    """
    Class to represent transitions from one activity to another,
    retains the transition as a tuple and saves the time span.
    """
    def __init__(self, name):
        """
        Receives a name and initializes the attributes of a new transition.
        """
        self._name = name
        self._weight = 0
        self._time = 0
        self._count = 0
        self._time_norm = 0
        self._weight_norm = 0


    def add(self, weight, time):
        """
        Receives a weight and a timestamp and adds them cumulatively
        to self._weight and self._time, respectively.
        Also adds the counter.
        """
        self._weight += weight
        self._time += time
        self._count += 1

    # def __hash__(self):
    #     return hash(tuple(sorted(self._name)))

    # def __eq__(self, other):
    #     return tuple(sorted(self._name)) == tuple(sorted(other._name))
