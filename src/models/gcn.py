"""Graph Graph Convolution Network Network (GCN) implementation for mineral deposit classification.

This module provides:
- MineralDepositGCN: A GCN model specialized for mineral deposit classification
  with customizable architecture including batch normalization, and a classification head.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MineralDepositGCN(nn.Module):
    """Graph Convolutional Network for mineral deposit classification.

    Attributes
    ----------
    n_classes : int
        Number of deposit classes to predict
    in_channels : int
        Number of input features
    hidden_channels : int
        Dimension of hidden layers
    improved : bool
        If True, use improved GCN aggregation
    add_self_loops : bool
        If True, add self-loops to the graph
    normalize : bool
        If True, normalize adjacency matrix
    n_layers : int
        Number of GCN layers
    dropout : float
        Dropout rate
    batch_norm : bool
        If True, apply batch normalization

    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        hidden_channels: int = 64,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = False,
        n_layers: int = 3,
        dropout: float = 0.3,
        batch_norm=True,
    ):
        """Initialize the Mineral Deposit GCN model.

        Parameters
        ----------
        n_classes : int
            Number of deposit classes to predict
        in_channels : int
            Number of input features
        hidden_channels : int, optional
            Dimension of hidden layers, by default 64
        improved : bool, optional
            If True, use improved GCN aggregation, by default False
        add_self_loops : bool, optional
            If True, add self-loops to the graph, by default True
        normalize : bool, optional
            If True, normalize adjacency matrix, by default False
        n_layers : int, optional
            Number of GCN layers, by default 3
        dropout : float, optional
            Dropout rate, by default 0.3
        batch_norm : bool, optional
            If True, apply batch normalization, by default True

        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Batch normalization layers
        # stabilize training and reduce internal covariate shift to ensure
        # that each layer receives inputs with a consistent distribution
        # BatchNorm1d for 2D sequence data (n_points, n_features)
        self.batch_norms = nn.ModuleList(
            [
                (nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity())
                for _ in range(n_layers)
            ]
        )
        # Multiple layers
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            GCNConv(
                in_channels,
                hidden_channels,
                improved=improved,
                add_self_loops=add_self_loops,
                normalize=normalize,
            )
        )  # in_channels => hidden_channels
        # Additional conv layers (hidden to hidden)
        for _ in range(n_layers - 1):
            self.convs.append(
                GCNConv(
                    hidden_channels,
                    hidden_channels,
                    improved=improved,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                )  # hidden_channels => hidden_channels* num_heads
            )
        # MLP head for classification: hidden_channels => n_classes
        self.classify = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, data):
        """Forward pass of the GCN model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data containing node features (x),
            edge indices (edge_index), and edge weights (edge_weight)

        Returns
        -------
        torch.Tensor
            Class logits for each node in the graph

        """
        assert hasattr(data, "x") and data.x is not None, (
            "Input data must have node features 'x'"
        )
        assert hasattr(data, "edge_index") and data.edge_index is not None, (
            "Input data must have 'edge_index'"
        )
        assert hasattr(data, "edge_weight") and data.edge_weight is not None, (
            "Input data must have 'edge_weight'"
        )
        # Extract node features and graph structure
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # Process through GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            x = F.elu(x)
            x = self.dropout(x)
            x = self.batch_norms[i](x)
        x = self.classify(x)
        return x
