from evolutionary_algorithm import EvolParams, OptimResults


def one_fifth_method(evol_params: EvolParams, optim_results: OptimResults) -> float:
    if (
        optim_results.iterations[-1] % evol_params.deviation_change_frequency == 0
        and len(optim_results.iterations) != 1
    ):
        successes = 0
        for i in range(evol_params.deviation_change_frequency):
            if optim_results.values[-(i + 1)] < optim_results.values[-(i + 2)]:
                successes += 1
        success_rate = successes / evol_params.deviation_change_frequency
        if success_rate > 1 / 5:
            return evol_params.deviation * (1 + evol_params.deviation_change)
        else:
            return evol_params.deviation * (1 - evol_params.deviation_change)
    else:
        return evol_params.deviation
