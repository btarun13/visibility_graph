import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool
import networkx as nx
import pandas as pd

# from VisibilityGraph.network_build import visibility_graph,rolling_visibility_graphs,plot_visibility_graph,plot_network_graph

def generate_graph_embedding_pytorch_geometric(num_nodes, edge_index, feature_dimension, num_layers, pooling_method):
    """Generates graph embeddings using PyTorch Geometric's GCNConv."""

    # Create random node features (replace with your actual features)
    x = torch.randn(num_nodes, feature_dimension)

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)  # Batch tensor (important!)

    # Define the GCN model
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels, num_layers):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(feature_dimension, hidden_channels))  # Input layer
            for _ in range(num_layers - 2): # Hidden Layers
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, feature_dimension)) # Output layer
            self.relu = torch.nn.ReLU()

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = conv(x, edge_index)
                x = self.relu(x)  # Activation function
            x = self.convs[-1](x, edge_index) # Last layer without activation

            return x

    # model = GCN(hidden_channels=64, num_layers=num_layers)  # You can adjust hidden_channels
    # embeddings = model(data.x, data.edge_index)
    model = GCN(hidden_channels=feature_dimension, num_layers=num_layers)
    node_embeddings = model(data.x, data.edge_index)  # Get node embeddings first

    # Pooling to get a graph-level embedding
    if pooling_method == "mean":
        graph_embedding = global_mean_pool(node_embeddings, data.batch)  # Mean pooling
    elif pooling_method == "sum":
        graph_embedding = global_add_pool(node_embeddings, data.batch)  # Sum pooling
    elif pooling_method == "max":
        graph_embedding = global_max_pool(node_embeddings, data.batch)  # Max pooling
    else:
        raise ValueError("Invalid pooling method. Choose 'mean', 'sum', or 'max'.")

    return graph_embedding


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

### for feeding through the pyG pipeline
# def make_graph(time_series,window):
#     """Generates Visibility Graphs on a rolling window basis."""
#     num_windows = len(time_series) - window + 1
#     rolling_vgs = {}
#     for i in range(num_windows):
#         window = time_series[i:i + window]
#         rolling_vgs[i] = visibility_graph(window)
#     rolling_vgs = rolling_vgs
#     return rolling_vgs

def make_graph(time_series, window):
    num_windows = len(time_series) - window + 1
    rolling_vgs = {}
    for i in range(num_windows):
        # Rename the slice to avoid conflict with the window parameter
        ts_window = time_series[i:i + window]
        rolling_vgs[i] = visibility_graph(ts_window)
    return rolling_vgs

def make_embedding(time_series, window, feature_dimension, num_layers, pooling_method):
    vg = make_graph(time_series, window)
    graph = vg[0]
    num_nodes = window
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    graph_embedding = generate_graph_embedding_pytorch_geometric(num_nodes, edge_index, feature_dimension, num_layers, pooling_method)
    return graph_embedding



def df_tensor_representation(ds:list, window:int,
                             col_types:list, feature_dimension:int,
                             num_layers:int, pooling_method:str):
    
    data_section = ds
    # print(len(data_section))
    
    count = 0
    for j in data_section:
        emb_comb = []
        for i in col_types:
            # print(list(j[i]))

            emb = make_embedding(list(j[i]), window=window, feature_dimension = feature_dimension, num_layers = num_layers, pooling_method = pooling_method)  ##data_section train_x[0]
            
            emb_comb.append(emb)

        concatenated_tensor = torch.cat(emb_comb, dim=1).detach().numpy()
        # print(concatenated_tensor.shape)
        if count == 0:
            num_cols = concatenated_tensor.shape[1]
            df = pd.DataFrame(concatenated_tensor,columns=[f'emb_{i}' for i in range(num_cols)])
            # print(df.shape)
        else:
            df2 = pd.DataFrame(concatenated_tensor,columns=[f'emb_{i}' for i in range(num_cols)])
            df = pd.concat([df,df2],axis=0)
        count += 1

    del emb_comb, concatenated_tensor
    return df



