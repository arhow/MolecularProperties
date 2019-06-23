import numpy as np
import scipy
from scipy import stats

def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
        return func
    return decorate_func

# @set_property("is_input_series", True)
@set_property("fctype", "simple")
def rms(x):
    return np.sqrt(np.mean(np.power(x, 2)))

@set_property("fctype", "simple")
def rssq(x):
    return np.sum(np.power(x, 2))

@set_property("fctype", "simple")
def crest_factor(x):
    if np.sqrt(np.mean(x ** 2)) == 0:
        return -9999
    else:
        return np.max(np.abs(x)) / (np.sqrt(np.mean(x ** 2)))

@set_property("fctype", "simple")
def peak_peak_amp(x):
    return np.max(np.abs(x)) - np.min(np.abs(x))


@set_property("fctype", "simple")
def clearance_factor(x):
    if np.mean(np.sqrt(abs(x))) ** 2 == 0:
        return -9999
    else:
        return np.max(np.abs(x)) / (np.mean(np.sqrt(abs(x))) ** 2)

@set_property("fctype", "simple")
def impluse_indicator(x):
    if np.mean(abs(x)) == 0:
        return -9999
    else:
        return np.max(np.abs(x)) / (np.mean(abs(x)))

@set_property("fctype", "simple")
def impulse_factor(x):
    if np.sum(np.abs(x)) == 0:
        return -9999
    else:
        return peak_peak_amp(x) / 2 / (np.sum(np.abs(x)) / len(x))

@set_property("fctype", "simple")
def variance(x):
    return np.var(x)

@set_property("fctype", "simple")
def skewness(x):
    return stats.skew(x)

@set_property("fctype", "simple")
def kurtosis(x):
    return stats.kurtosis(x)

@set_property("fclvl", 0)
@set_property("fctype", "simple")
def mean(x):
    return np.mean(x)

@set_property("fctype", "simple")
def median(x):
    return np.median(x)

@set_property("fclvl", 0)
@set_property("fctype", "simple")
def max(x):
    return np.max(x)

@set_property("fclvl", 0)
@set_property("fctype", "simple")
def min(x):
    return np.min(x)

@set_property("fclvl", 0)
@set_property("fctype", "simple")
def std(x):
    return np.std(x)

@set_property("fctype", "simple")
def quantile99(x):
    return np.quantile(x, 0.99)

@set_property("fctype", "simple")
def quantile95(x):
    return np.quantile(x, 0.95)

@set_property("fctype", "simple")
def quantile75(x):
    return np.quantile(x, 0.75)

@set_property("fctype", "simple")
def quantile25(x):
    return np.quantile(x, 0.25)

@set_property("fctype", "simple")
def quantile05(x):
    return np.quantile(x, 0.05)

@set_property("fctype", "simple")
def quantile01(x):
    return np.quantile(x, 0.01)

@set_property("fctype", "simple")
def abs_quantile99(x):
    return np.quantile(np.abs(x), 0.99)

@set_property("fctype", "simple")
def abs_quantile95(x):
    return np.quantile(np.abs(x), 0.95)

@set_property("fctype", "simple")
def abs_quantile75(x):
    return np.quantile(np.abs(x), 0.75)

@set_property("fctype", "simple")
def abs_quantile25(x):
    return np.quantile(np.abs(x), 0.25)

@set_property("fctype", "simple")
def abs_quantile05(x):
    return np.quantile(np.abs(x), 0.05)

@set_property("fctype", "simple")
def abs_quantile01(x):
    return np.quantile(np.abs(x), 0.01)

@set_property("fctype", "simple")
def entropy(x):
    return stats.entropy(np.abs(x))

@set_property("fctype", "simple")
def form_factor(x):
    if np.mean(np.abs(x)) == 0:
        return -9999
    else:
        return rms(x) / np.mean(np.abs(x))

@set_property("fctype", "simple")
def no_zero_crossing(x):
    return (np.diff(np.sign(x)) != 0).sum()

@set_property("fctype", "simple")
def peak_to_average_power_ratio(x):
    return np.power(crest_factor(x), 2)

@set_property("fctype", "simple")
def smoothness(x):
    return 1 - 1 / (1 + np.power(std(x), 2))


