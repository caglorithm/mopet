class MopetException(Exception):
    """ Base Exception for all exceptions thrown by mopet library. """

    pass


class ExplorationNotFoundError(MopetException):
    """ Thrown if exploration could not be found. """

    pass


class Hdf5FileNotExistsError(MopetException):
    """ Thrown if HDF5 file does not exist. """

    pass
