import numpy as np
from inspect import getargspec
from utilities import aggregate_feature_calculators
import warnings

string_types = str
integer_types = int
class_types = type
text_type = str
binary_type = bytes

class ComprehensiveFCParameters(dict):

    def __init__(self):
        name_to_param = {}

        for name, func in aggregate_feature_calculators.__dict__.items():
            if callable(func) and hasattr(func, "fctype") and len(getargspec(func).args) == 1:
                name_to_param[name] = None

        name_to_param.update({
            # "rms": [],
            # "rssq": [],
            # "crest_factor": [],
            # "peak_peak_amp": [],
        })

        super(ComprehensiveFCParameters, self).__init__(name_to_param)


def _convert_to_output_format(param):

    def add_parenthesis_if_string_value(x):
        if isinstance(x, string_types):
            return '"' + str(x) + '"'
        else:
            return str(x)

    return "__".join(str(key) + "_" + add_parenthesis_if_string_value(param[key]) for key in sorted(param.keys()))


def aggregate_features(x, fc_parameters=ComprehensiveFCParameters()):

    def _f():
        for function_name, parameter_list in fc_parameters.items():
            func = getattr(aggregate_feature_calculators, function_name)

            # If the function uses the index, pass is at as a pandas Series.
            # Otherwise, convert to numpy array
            if hasattr(func, 'is_input_series'):
                x_ = x.values
            else:
                x_ = x

            if func.fctype == "combiner":
                result = func(x_, param=parameter_list)
            else:
                if parameter_list:
                    result = [(_convert_to_output_format(param), func(x_, **param)) for param in parameter_list]
                else:
                    result = [("", func(x_))]

            for key, item in result:
                yield {"function_name": function_name, "param": key, "result": item}

    return list(_f())

"""
x = np.arange(1985)
a = do_the_shit(x)
print('finished')
"""