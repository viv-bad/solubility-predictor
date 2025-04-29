import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree

class GNNLayer(MessagePassing):
    """
    A custom Graph Neural Network layer with message passing.
    """

    def __init__(self, in_channels, out_channels):
        # aggr will define how messages are aggregated (sum, mean, max etc)
        super(GNNLayer, self).__init__(aggr='add')

        # neural networks for transforming node and edge features
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + 12, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Update function for node features after message passing
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the GNN layer.
        """

        # Add self loops to the adjacency matrix
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr = edge_attr, fill_value=torch.zeros(1, edge_attr.shape[1], device=edge_index.device),
            num_nodes = x.shape[0]
        )

        # start propagating the messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        #x_i: features of target nodes (shape: [num_edges, in_channels])
        #x_j: features of source nodes (shape: [num_edges, in_channels])
        #edge_attr: features of edges (shape: [num_edges, edge_feature_dim])

        # combine the source node features with edge features
        edge_features = torch.cat([x_j, edge_attr], dim=1)

        # transform the edge features
        edge_features = self.edge_mlp(edge_features)

        # Return the message
        return edge_features
    
    def update(self, aggr_out, x):
        # aggr_out: aggregated messages (shape: [num_nodes, out_channels])
        # x: original node features (shape: [num_nodes, in_channels])

        # combine the aggregated messages with origiinal node features
        node_features = torch.cat([x, aggr_out], dim = 1)

        # Update node features
        node_features = self.update_mlp(node_features)

        return node_features


class SolubilityGNN(nn.Module):
    """
    Graph Neural Network for solubility prediction.
    """

    def __init__(self, node_feature_dim = 160, edge_feature_dim = 12, hidden_dim = 64, num_layers=3, dropout=0.2):
        super(SolubilityGNN, self).__init__()

        # embedding layer to reduce dimensionality of node features
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)

        # stack of GNN layers for message passing
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GNNLayer(hidden_dim, hidden_dim))

        for _ in range(num_layers - 1):
            self.gnn_layers.append(GNNLayer(hidden_dim, hidden_dim))
        
        # Readout and prediction layers
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), 
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # predict single value output - solubility
        )


    def forward(self, data):
        # Get node features, edge indices, and edge features from data
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # initial embedding of node features
        x = self.node_embedding(x)

        # apply GNN layers for message passing
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)

            # after each layer, implement non linearity
            x = F.relu(x)

        # global pooling: combine node features for each graph in the batch
        # to get a single feature vector per molecule
        x = global_mean_pool(x, batch)

        # final pred
        solubility = self.readout_mlp(x)

        return solubility
        
        
        
