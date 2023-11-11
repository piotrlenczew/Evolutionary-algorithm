import numpy as np
import random
from typing import Optional, Callable, Any
from stop_map import evol_stop_map


#warunki stopu przez n iteracji średnia nie zmieniła się o jakąś tolerance
#przez n iteracji nie było sukcesu
#wrzucanie jako parametru funkcji metody deviation_adaptation.py np. metody 1/5
#jupiter albo latex

class EvolParams:
    def __init__(
        self,
        deviation: Optional[float] = None,
        deviation_change: Optional[float] = None,
        deviation_change_frequency: Optional[int] = None,
        max_iter: Optional[int] = None,
        tolerance: Optional[float] = None,
    ):
        if deviation:
            self.deviation = deviation
        else:
            self.deviation = 1
        if deviation_change:
            self.deviation_change = deviation_change
        else:
            self.deviation_change = 0.2
        if deviation_change_frequency:
            self.deviation_change_frequency = deviation_change_frequency
        else:
            self.deviation_change_frequency = 10
        if max_iter:
            self.max_iter = max_iter
        else:
            self.max_iter = 500
        if tolerance:
            self.tolerance = tolerance
        else:
            self.tolerance = 1e-6


class EvolResults:
    def __init__(self, iterations: [int], values: [float], reason_for_stop: Optional[str] = None):
        self.iterations = iterations
        self.values = values
        self.reason_for_stop = reason_for_stop


class EvolStopConditions: #evol_results jest podawane w conditions_not_met, ponieważ musi być aktualne
    def __init__(self, evol_params: EvolParams, evol_stop_map):
        self.evol_params = evol_params
        self.evol_stop_map = evol_stop_map
        self.reason_for_stop = None

    def conditions_not_meet(self, evol_results: EvolResults):
        for stop_name, stop in self.evol_stop_map.items():
            if stop(self.evol_params, evol_results):
                self.reason_for_stop = stop_name
                return False
        return True


def evolutionary_algorithm(f: Callable, x0: Any, deviation_adaptation_method, evol_params: EvolParams) -> EvolResults:
    stop_conditions = EvolStopConditions(evol_params, evol_stop_map)
    results = EvolResults([0], [f(x0).item()])
    current_x = np.array(x0)

    i = 0
    while stop_conditions.conditions_not_meet(results):
        i += 1
        next_x = np.array([random.gauss(evol_params.deviation, a) for a in current_x])
        if f(next_x) < f(current_x):
            current_x = next_x
        evol_params.deviation = deviation_adaptation_method(evol_params, results)
        results.iterations.append(i)
        results.values.append((f(current_x)).item())
    results.reason_for_stop = stop_conditions.reason_for_stop
    return results
