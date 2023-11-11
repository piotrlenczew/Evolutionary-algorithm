evol_stop_map = {
    "max_iter": lambda evol_params, evol_results: evol_results.iterations[-1] >= evol_params.max_iter,

}