import matplotlib.pyplot as plt
from evolutionary_algorithm import OptimResults


def plot_convergence(data_list: [OptimResults], labels: [str], title: str):
    plt.figure(figsize=(10, 6))
    graph_colors = ["b", "g", "r", "c", "m", "y"]

    for i, data in enumerate(data_list):
        iterations = data.iterations
        values = data.values
        plt.plot(
            iterations,
            values,
            # marker="o",
            linestyle="-",
            color=graph_colors[i % len(graph_colors)],
            label=labels[i],
        )

    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.title(title)
    #plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()
