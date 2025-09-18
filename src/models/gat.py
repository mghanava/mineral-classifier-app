"""Graph Attention Network (GAT) implementation for mineral deposit classification.

This module provides:
- MineralDepositGAT: A GAT model specialized for mineral deposit classification
  with customizable architecture including multi-head attention, batch normalization, and a classification head.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MineralDepositGAT(nn.Module):
    """Graph Attention Network for mineral deposit classification.

    This class implements a GAT model specialized for classifying mineral deposits
    using graph-structured data. It supports multi-head attention, batch normalization,
    and includes a classification head.

    Parameters
    ----------
    n_classes : int
        Number of output classes for classification
    in_channels : int
        Number of input features
    hidden_channels : int, optional
        Dimension of hidden layers (default: 64)
    n_layers : int, optional
        Number of GAT layers (default: 3)
    n_heads : int, optional
        Number of attention heads (default: 5)
    dropout : float, optional
        Dropout rate (default: 0.3)
    negative_slope : float, optional
        LeakyReLU negative slope (default: 0.2)
    add_self_loops : bool, optional
        If True, add self-loops to the graph (default: True)
    bias : bool, optional
        If True, add bias to linear layers (default: True)
    residual : bool, optional
        If True, add residual connections (default: False)
    batch_norm : bool, optional
        If True, apply batch normalization (default: True)

    """

    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 3,
        n_heads: int = 5,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True,
        residual: bool = False,
        batch_norm: bool = True,
    ):
        """Initialize the GAT model for mineral deposit classification.

        Parameters
        ----------
        n_classes : int
            Number of output classes for classification
        in_channels : int
            Number of input features
        hidden_channels : int, optional
            Dimension of hidden layers, by default 64
        n_layers : int, optional
            Number of GAT layers, by default 3
        n_heads : int, optional
            Number of attention heads, by default 5
        dropout : float, optional
            Dropout rate, by default 0.3
        negative_slope : float, optional
            LeakyReLU negative slope, by default 0.2
        add_self_loops : bool, optional
            If True, add self-loops to the graph, by default False
        bias : bool, optional
            If True, add bias to linear layers, by default True
        residual : bool, optional
            If True, add residual connections, by default False
        batch_norm : bool, optional
            If True, apply batch normalization, by default True

        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Batch normalization layers
        self.batch_norms = nn.ModuleList(
            [
                (
                    nn.BatchNorm1d(hidden_channels * n_heads)
                    if batch_norm
                    else nn.Identity()
                )
                for _ in range(n_layers - 1)
            ]
        )
        # batch norm on last layer
        self.batch_norms.append(
            nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity()
        )
        # Multiple layers
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=n_heads,
                add_self_loops=add_self_loops,
                negative_slope=negative_slope,
                bias=bias,
                residual=residual,
                dropout=dropout,
            )
        )  # in_channels => hidden_channels
        # Middle layers
        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * n_heads,
                    hidden_channels,
                    heads=n_heads,
                    add_self_loops=add_self_loops,
                    negative_slope=negative_slope,
                    bias=bias,
                    residual=residual,
                    dropout=dropout,
                )  # hidden_channels * n_heads => hidden_channels* n_heads
            )
        # Last layer
        self.convs.append(
            GATConv(
                hidden_channels * n_heads,
                hidden_channels,
                heads=1,
                add_self_loops=add_self_loops,
                negative_slope=negative_slope,
                bias=bias,
                residual=residual,
                dropout=dropout,
            )  # hidden_channels * n_heads => hidden_channels
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
        """Forward pass of the GAT model.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data containing node features (x),
            edge indices (edge_index), and edge attributes (edge_attr)

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
        assert hasattr(data, "edge_attr") and data.edge_attr is not None, (
            "Input data must have 'edge_attr'"
        )
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Process through GAT layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.dropout(x)
            x = self.batch_norms[i](x)
        x = self.classify(x)
        return x
