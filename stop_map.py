n = 200
m = 100

evol_stop_map = {
    "max iter": lambda evol_params, evol_results: evol_results.iterations[-1] >= evol_params.max_iter,
    f"low avg improvement in {n} last iter": lambda evol_params, evol_results: evol_results.iterations[-1] > n and evol_results.values[-n] - (sum(evol_results.values[-n:]) / n) < evol_params.tolerance,
    f"No improvement in {m} last iter": lambda evol_params, evol_results: evol_results.iterations[-1] > m and evol_results.values[-m] <= evol_results.values[-1]
}
