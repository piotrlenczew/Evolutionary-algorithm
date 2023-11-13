from cec2017.simple import f1, f9
from evolutionary_algorithm import one_plus_one_algorithm, EvolParams, OptimResults
from Gradient_descent import gradient_descent, GradParam
from deviation_adaptation import one_fifth_method
from Plotting import plot_convergence
from stop_map import evol_stop_map
import numpy as np


repetition_nr = 15
x = np.array([100] * 10)
functions = {"f1": f1, "f9": f9}
deviations = [100]


def avg_result(results: [OptimResults]):
    zipped_values = zip(
        *[result.values for result in results]
    )  # * passes multiple objects as an individual one
    averages = [sum(values) / len(values) for values in zipped_values]
    return OptimResults(results[1].iterations, averages, None)


for name, function in functions.items():
    average_results = []
    average_results_labels = []
    for deviation in deviations:
        results = []
        labels = []
        print(f"Repetitions of minimizing {name} function")
        for repetition in range(repetition_nr):
            evol_result = one_plus_one_algorithm(
                function, x, one_fifth_method, evol_stop_map, EvolParams(deviation)
            )
            results.append(evol_result)
            labels.append(f"Repetition nr {repetition+1}")
            print(
                f"Reason for stop in repetition nr {repetition+1}: {evol_result.reason_for_stop}\nReached value: {evol_result.values[-1]}"
            )
        plot_convergence(
            results,
            labels,
            f"Graph for {repetition_nr} repetitions of {name} function and deviation = {deviation}.",
        )
        if all(len(result.values) == len(results[0].values) for result in results):
            average_results.append(avg_result(results))
            average_results_labels.append(f"{name} dev={deviation}")
            plot_convergence(
                [avg_result(results)],
                [f"{name}"],
                f"Graph for average of {name} function and deviation = {deviation}.",
            )
        else:
            print(
                f"\nCannot create average graph for {name}. Different lengths of value lists.\n"
            )
        print(
            f"Average result reached for one plus one: {sum(result.values[-1] for result in results)/repetition_nr}"
        )
        grad_results = gradient_descent(function, x, GradParam())
        print(f"Result reached for gradient descent: {grad_results.values[-1]}")
        plot_convergence(
            [grad_results],
            [f"grad {name}"],
            f"Graph for gradient descent minimalisation of {name} function.",
        )
        print("\n")
    if len(average_results) > 0:
        plot_convergence(
            average_results,
            average_results_labels,
            f"Graph for average of {name} function for multiple deviations.",
        )
