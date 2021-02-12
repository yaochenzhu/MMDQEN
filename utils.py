import numpy  as np
import pandas as pd
from scipy.interpolate import interp1d

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class ConstantSchedule():
    def __init__(self, value):
        """Value remains constant over time.
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

class PiecewiseSchedule():
    def __init__(self, 
                 endpoints, 
                 interpolation=linear_interpolation, 
                 outside_value=None):
        """
        Piecewise schedule.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule():
    def __init__(self, 
                 schedule_timesteps, 
                 final_p, 
                 initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExponentialSchedule():
    def __init__(self, half_value):
        self.alpha = -(np.log(0.5) / half_value)

    def value(self, t):
        return np.exp(-self.alpha*t)