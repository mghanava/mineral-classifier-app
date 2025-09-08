import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MineralDepositGCN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        in_channels: int,
        hidden_channels: int = 64,
        improved: bool = False,
        add_self_loops: bool = False,
        normalize: bool = False,
        n_layers: int = 3,
        dropout: float = 0.3,
        batch_norm=True,
    ):
        super().__init__()
        # prevent overfitting
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
        # extract hidden features in graph
        # Multiple GAT layers
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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Process through GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.batch_norms[i](x)
        x = self.classify(x)
        return x
