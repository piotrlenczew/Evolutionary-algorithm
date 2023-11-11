from evolutionary_algorithm import EvolParams, EvolResults


def one_fifth_method(evol_params: EvolParams, evol_results: EvolResults):
    if evol_results.iterations[-1] % evol_params.deviation_change_frequency == 0 and len(evol_results.iterations) != 1:
        successes = 0
        for i in range(evol_params.deviation_change_frequency):
            if evol_results.values[-(i+1)] < evol_results.values[-(i+2)]:
                successes += 1
        success_rate = successes/evol_params.deviation_change_frequency
        if success_rate > 1/5:
            return evol_params.deviation * (1 + evol_params.deviation_change)
        else:
            return evol_params.deviation * (1 - evol_params.deviation_change)
    else:
        return evol_params.deviation

