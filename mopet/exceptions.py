class MopetException(Exception):
    """ Base Exception for all exceptions thrown by mopet library. """

    pass


class ExplorationNotFoundError(MopetException):
    """ Thrown if exploration could not be found in HDF5 file. """

    pass


class Hdf5FileNotExistsError(MopetException):
    """ Thrown if HDF5 file does not exist. """

    pass


class ExplorationExistsError(MopetException):
    """ Thrown if the exploration name exists already as a group in the HDF5 file. """

    pass
