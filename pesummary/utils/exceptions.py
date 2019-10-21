class InputError(Exception):
    """
    """
    def __init__(self, message):
        super(InputError, self).__init__(message)


class PlotError(Exception):
    """
    """
    def __init__(self, message):
        super(PlotError, self).__init__(message)
