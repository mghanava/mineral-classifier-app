import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class MineralDepositGAT(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 3,
        n_heads: int = 5,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        add_self_loops: bool = False,
        bias: bool = True,
        residual: bool = False,
        batch_norm: bool = True,
    ):
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
        # Multiple GAT layers
        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=n_heads,
                negative_slope=negative_slope,
                bias=bias,
                residual=residual,
                add_self_loops=add_self_loops,
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
                add_self_loops=False,
                dropout=dropout,
            )  # hidden_channels * n_heads => hidden_channels
        )
        # MLP head for classification: hidden_channels => n_classes
        self.classify = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Process through GAT layers
        for i, conv in enumerate(self.convs):
            # Apply GAT convolution
            x = conv(x, edge_index, edge_attr)
            x = F.elu(x)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
        x = self.classify(x)
        return x
