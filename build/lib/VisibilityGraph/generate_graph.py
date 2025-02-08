import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, global_max_pool

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


def make_embedding(time_series, window, feature_dimension, num_layers, pooling_method):
    vg = make_graph(time_series, window)
    graph = vg[0]
    num_nodes = window
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    graph_embedding = generate_graph_embedding_pytorch_geometric(num_nodes, edge_index, feature_dimension, num_layers, pooling_method)
    return graph_embedding


### for feeding through the pyG pipeline
def make_graph(time_series,window):
    rolling_vgs = rolling_visibility_graphs(time_series, window)
    return rolling_vgs



def df_tensor_representation(ds:list, window:int,
                             col_types:list, feature_dimension:int,
                             num_layers:int, pooling_method:str):
    
    data_section = ds
    # col_types = ['high', 'open', 'close', 'low', 'amount', 'vol']
    emb_comb = []
    for i in col_types:
        emb = make_embedding(data_section[i], window=window, feature_dimension, num_layers, pooling_method)  ##data_section train_x[0]
        emb_comb.append(emb)
    concatenated_tensor = torch.cat(emb_comb, dim=1)
    return concatenated_tensor

