import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visibility_graph(time_series):
    """Constructs a Visibility Graph (VG) from a time series window."""
    G = nx.Graph()
    n = len(time_series)

    for i in range(n):
        G.add_node(i, value=time_series[i])

    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                if time_series[k] > time_series[i] + (time_series[j] - time_series[i]) * (k - i) / (j - i):
                    visible = False
                    break
            if visible:
                G.add_edge(i, j)

    return G

def rolling_visibility_graphs(time_series, window_size):
    """Generates Visibility Graphs on a rolling window basis."""
    num_windows = len(time_series) - window_size + 1
    rolling_vgs = {}

    for i in range(num_windows):
        window = time_series[i:i + window_size]
        rolling_vgs[i] = visibility_graph(window)

    return rolling_vgs

def plot_visibility_graph(time_series, G, window_start, window_size):
    """Plots the time series with visibility edges for a rolling window."""
    plt.figure(figsize=(10, 5))
    x = np.arange(window_start, window_start + window_size)
    y = time_series[window_start:window_start + window_size]

    # Plot time series in gray
    plt.bar(range(1, len(time_series) + 1), time_series, color='gray', alpha=0.3)

    # Highlight rolling window
    plt.bar(x + 1, y, color='blue', alpha=0.6)

    plot = []
    # Draw visibility edges for this window
    for i, j in G.edges():
        a = plt.plot([x[i] + 1, x[j] + 1], [y[i], y[j]], 'k-', alpha=0.7)
        plot.append(a)
        plt.plot([x[i] + 1, x[j] + 1], [y[i], y[j]], 'k-', alpha=0.7)

    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Visibility Graph for Rolling Window [{window_start}:{window_start + window_size}]")
    plt.grid(True)
    plt.show()

    return plot

def plot_network_graph(G, time_series, window_start):
    """Plots the Visibility Graph as a network for a rolling window."""
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [time_series[window_start + i] * 100 for i in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_sizes, cmap=plt.cm.Blues, edge_color="black")
    plt.title(f"Network Graph for Rolling Window [{window_start}:{window_start + len(G.nodes())}]")
    plt.show()

