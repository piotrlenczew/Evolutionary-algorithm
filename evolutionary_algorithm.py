import numpy as np
import random
from typing import Optional, Callable, Any

#stop_map = {}
#stop_map["stop_cond"] = lambda optim_params, optim_solutions: return optim_solutions.iter < ...

#if stop_map["stop_cond"](optim_params, optim_solutions):

#class StopConditions(GradParams, GradResults): #GraadResults musi być w aktualnym momencie
    # metoda if_stop która przejdzie przez stop_map i sprawdzi czy któryś z warunków stopu został użyty

#warunki stopu przez n iteracji średnia nie zmieniła się o jakąś tolerance
#przez n iteracji nie było sukcesu
#wrzucanie jako parametru funkcji metody step_size_adaptation np. metody 1/5
#jupiter albo latex

class EvolParam:
    def __init__(
        self,
        deviation: Optional[float] = None,
        max_iter: Optional[int] = None,
        #tolerance: Optional[float] = None,
    ):
        if deviation:
            self.deviation = deviation
        else:
            self.deviation = 0.01
        if max_iter:
            self.max_iter = max_iter
        else:
            self.max_iter = 100
        # if tolerance:
        #     self.tolerance = tolerance
        # else:
        #     self.tolerance = 1e-6


class EvolResults:
    def __init__(self, iterations: [int], values: [float], reason_for_stop: Optional[str]=None):
        self.iterations = iterations
        self.values = values
        #self.reason_for_stop = reason_for_stop


def evolutionary_algorithm(f: Callable, x0: Any, evol_param: EvolParam) -> EvolResults:
    results = EvolResults([0], [f(x0)])
    current_x = np.array(x0)

    i = 0
    while i <= evol_param.max_iter:
        i += 1
        next_x = np.array([random.gauss(evol_param.deviation, a) for a in current_x])
        if f(next_x) < f(current_x):
            current_x = next_x
        results.iterations.append(i)
        results.values.append((f(current_x)))
    return results
