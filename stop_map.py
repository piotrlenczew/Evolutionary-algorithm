n = 250
m = 200

evol_stop_map = {
    "max iter": lambda evol_params, optim_results:
    optim_results.iterations[-1] >= evol_params.max_iter,

    f"low avg improvement in {n} last iter": lambda evol_params, optim_results:
    optim_results.iterations[-1] > n
    and optim_results.values[-n] - (sum(optim_results.values[-n:]) / n) < evol_params.tolerance,

    f"No improvement in {m} last iter": lambda evol_params, optim_results:
    optim_results.iterations[-1] > m
    and optim_results.values[-m] <= optim_results.values[-1],
}
