import json
import numpy as np

def parse_kwargs(param, start_with='', exception_list=['']):
    """ Parse dictionary by starting with string

    Parameters
    ----------
    param : dict
        target dictionary to parse
    start_with : str
        target parser Defaults to ''.
    exception_list : list
        list of keys to exclude. Defaults to [''].

    Returns
    -------
    dict
        parsed key-values 
    """
    kwargs = {}
    for _key, _value in param.items():
        if _key.startswith(start_with) and (_key not in exception_list):
            kwargs[_key.replace(start_with, '')] = _value
    return kwargs

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return super(CustomJSONEncoder, self).encode(bool(obj))
        return super(CustomJSONEncoder, self).default(self, obj)