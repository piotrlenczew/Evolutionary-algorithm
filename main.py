from cec2017.functions import f1, f9, f2
from evolutionary_algorithm import evolutionary_algorithm, EvolParams, EvolResults
from deviation_adaptation import one_fifth_method
from Plotting import plot_convergence
import numpy as np


def wrapper_1d_2d(x_1d):
    return x_1d.reshape(1, -1)


def F1(x):
    return(f1(wrapper_1d_2d(x)))

def F9(x):
    return(f9(wrapper_1d_2d(x)))

x = np.array([100] * 10)
result = evolutionary_algorithm(F9, x, one_fifth_method, EvolParams())
print(result.values)
print(result.iterations)
print(result.reason_for_stop)

plot_convergence([result], ["F9"], "Graph for F9 function")
