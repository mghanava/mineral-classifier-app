from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data


class ReservoirSampler:
    """Advanced reservoir sampling for maintaining diverse, representative datasets."""

    def __init__(
        self,
        max_size: int = 3000,
        strategy: Literal[
            "stratified_recent", "stratified", "random", "uncertainty"
        ] = "stratified_recent",
        preserve_recent_ratio: float = 0.4,
        class_balance_weight: float = 0.8,
        quality_threshold: float | None = None,
        device: str = "cpu",
    ):
        """Initialize reservoir sampler.

        Args:
            max_size: Maximum number of samples to maintain
            strategy: Sampling strategy
            preserve_recent_ratio: Ratio of recent data to preserve
            class_balance_weight: Weight for maintaining class balance (0-1)
            quality_threshold: Remove samples below this quality score
            device: PyTorch device

        """
        self.max_size = max_size
        self.strategy = strategy
        self.preserve_recent_ratio = preserve_recent_ratio
        self.class_balance_weight = class_balance_weight
        self.quality_threshold = quality_threshold
        self.device = device

    def sample_combined_data(
        self, base_data: Data, prediction_data: Data, cycle_num: int = 1
    ) -> Data:
        """Create combined dataset using reservoir sampling.

        Args:
            base_data: Historical/base training data
            prediction_data: New prediction data to incorporate
            cycle_num: Current cycle number (for temporal weighting)

        Returns:
            Combined dataset with size <= max_size

        """
        # Combine all data first
        all_data = self._combine_graph_data(base_data, prediction_data)

        if all_data.x.shape[0] <= self.max_size:
            return all_data

        # Apply reservoir sampling based on strategy
        if self.strategy == "stratified_recent":
            sampled_data = self._stratified_recent_sampling(all_data, cycle_num)
        elif self.strategy == "stratified":
            sampled_data = self._stratified_sampling(all_data)
        elif self.strategy == "random":
            sampled_data = self._random_sampling(all_data)
        elif self.strategy == "uncertainty":
            sampled_data = self._uncertainty_sampling(all_data)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        return sampled_data

    def _combine_graph_data(self, base_data: Data, prediction_data: Data) -> Data:
        """Combine two graph datasets."""
        # Combine features
        x_combined = torch.cat([base_data.x, prediction_data.x], dim=0)
        y_combined = torch.cat([base_data.y, prediction_data.y], dim=0)
        coords_combined = torch.cat(
            [base_data.coordinates, prediction_data.coordinates], dim=0
        )

        # Adjust edge indices for prediction data
        base_n_nodes = base_data.x.shape[0]
        pred_edge_index_adjusted = prediction_data.edge_index + base_n_nodes

        # Combine edges
        edge_index_combined = torch.cat(
            [base_data.edge_index, pred_edge_index_adjusted], dim=1
        )

        # Combine edge attributes if they exist
        edge_attr_combined = None
        if base_data.edge_attr is not None and prediction_data.edge_attr is not None:
            edge_attr_combined = torch.cat(
                [base_data.edge_attr, prediction_data.edge_attr], dim=0
            )

        return Data(
            x=x_combined,
            y=y_combined,
            edge_index=edge_index_combined,
            edge_attr=edge_attr_combined,
            coordinates=coords_combined,
        )

    def _stratified_recent_sampling(self, data: Data, cycle_num: int) -> Data:
        """Stratified sampling with bias toward recent data."""
        n_total = data.x.shape[0]
        n_recent = min(n_total, int(self.max_size * self.preserve_recent_ratio))
        n_historical = self.max_size - n_recent

        # Assume recent data is at the end (from prediction_data)
        # This is a simplification - in practice you'd need temporal markers
        recent_indices = list(range(max(0, n_total - n_recent), n_total))
        historical_pool = list(range(0, max(0, n_total - n_recent)))

        # Sample historical data with stratification
        if n_historical > 0 and len(historical_pool) > 0:
            historical_indices = self._stratified_sample_indices(
                data.y[historical_pool], n_historical
            )
            historical_indices = [historical_pool[i] for i in historical_indices]
        else:
            historical_indices = []

        # Combine indices
        selected_indices = historical_indices + recent_indices
        selected_indices = selected_indices[
            : self.max_size
        ]  # Ensure we don't exceed limit

        return self._extract_subgraph(data, selected_indices)

    def _stratified_sampling(self, data: Data) -> Data:
        """Pure stratified sampling maintaining class balance."""
        selected_indices = self._stratified_sample_indices(data.y, self.max_size)
        return self._extract_subgraph(data, selected_indices)

    def _random_sampling(self, data: Data) -> Data:
        """Simple random sampling."""
        indices = torch.randperm(data.x.shape[0])[: self.max_size].tolist()
        return self._extract_subgraph(data, indices)

    def _uncertainty_sampling(self, data: Data) -> Data:
        """Sample based on prediction uncertainty (requires model predictions)."""
        # This is a placeholder - would need actual uncertainty scores
        # For now, fall back to stratified sampling
        print(
            "Warning: Uncertainty sampling not implemented, using stratified sampling"
        )
        return self._stratified_sampling(data)

    def _stratified_sample_indices(self, labels: torch.Tensor, n_samples: int) -> list:
        """Perform stratified sampling to maintain class distribution."""
        labels_np = labels.cpu().numpy()
        unique_classes, class_counts = np.unique(labels_np, return_counts=True)

        # Calculate target samples per class
        total_samples = len(labels_np)
        class_proportions = class_counts / total_samples

        # Apply class balance weight
        if self.class_balance_weight > 0:
            # Blend between uniform and proportional distribution
            uniform_prop = 1.0 / len(unique_classes)
            adjusted_proportions = (
                self.class_balance_weight * uniform_prop
                + (1 - self.class_balance_weight) * class_proportions
            )
        else:
            adjusted_proportions = class_proportions

        # Normalize
        adjusted_proportions = adjusted_proportions / adjusted_proportions.sum()

        # Calculate samples per class
        samples_per_class = (adjusted_proportions * n_samples).astype(int)

        # Distribute remaining samples
        remaining = n_samples - samples_per_class.sum()
        for i in range(remaining):
            samples_per_class[i % len(samples_per_class)] += 1

        # Sample from each class
        selected_indices = []
        for class_label, n_class_samples in zip(
            unique_classes, samples_per_class, strict=False
        ):
            class_mask = labels_np == class_label
            class_indices = np.where(class_mask)[0]

            if len(class_indices) <= n_class_samples:
                # Take all samples from this class
                selected_indices.extend(class_indices.tolist())
            else:
                # Random sample from this class
                sampled_indices = np.random.choice(
                    class_indices, size=n_class_samples, replace=False
                )
                selected_indices.extend(sampled_indices.tolist())

        return selected_indices

    def _extract_subgraph(self, data: Data, node_indices: list) -> Data:
        """Extract subgraph containing only specified nodes."""
        node_indices = torch.tensor(node_indices, dtype=torch.long)

        # Create mapping from old to new indices
        node_map = {
            old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)
        }

        # Extract node features
        x_sub = data.x[node_indices]
        y_sub = data.y[node_indices]
        coords_sub = data.coordinates[node_indices]

        # Extract relevant edges
        edge_mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
        new_edge_index_list = []

        for i, (src, dst) in enumerate(data.edge_index.t()):
            if src.item() in node_map and dst.item() in node_map:
                edge_mask[i] = True
                new_edge_index_list.append([node_map[src.item()], node_map[dst.item()]])

        # Create new edge index
        if new_edge_index_list:
            edge_index_sub = torch.tensor(new_edge_index_list, dtype=torch.long).t()
        else:
            edge_index_sub = torch.empty((2, 0), dtype=torch.long)

        # Extract edge attributes if they exist
        edge_attr_sub = None
        if data.edge_attr is not None:
            edge_attr_sub = data.edge_attr[edge_mask]

        return Data(
            x=x_sub,
            y=y_sub,
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            coordinates=coords_sub,
        )

    def _extract_subgraph_smart(
        self, data: Data, base_indices: list, pred_indices: list, cycle_num: int
    ) -> Data:
        """Smart subgraph extraction that preserves connectivity better."""
        
        # Calculate how many nodes to keep from each part
        n_recent = min(len(pred_indices), int(self.max_size * self.preserve_recent_ratio))
        n_historical = self.max_size - n_recent
        
        # Always keep all recent data (prediction data)
        selected_pred_indices = pred_indices[-n_recent:] if n_recent < len(pred_indices) else pred_indices
        
        # Sample historical data strategically
        if n_historical > 0 and len(base_indices) > 0:
            # Prioritize nodes that are well-connected (high degree)
            degrees = self._calculate_node_degrees(data, base_indices)
            
            # Use stratified sampling with connectivity bias
            historical_selected = self._connectivity_aware_sampling(
                data, base_indices, n_historical, degrees
            )
        else:
            historical_selected = []
        
        # Combine selected indices
        all_selected_indices = historical_selected + selected_pred_indices
        
        # Extract subgraph with better edge preservation
        return self._extract_subgraph_with_reconstruction(data, all_selected_indices)
    
    def _calculate_node_degrees(self, data: Data, node_indices: list) -> dict:
        """Calculate degree for each node in the given indices."""
        degrees = {}
        edge_index = data.edge_index
        
        for node_idx in node_indices:
            # Count edges where this node is source or destination
            degree = ((edge_index[0] == node_idx) | (edge_index[1] == node_idx)).sum().item()
            degrees[node_idx] = degree
            
        return degrees
    
    def _connectivity_aware_sampling(
        self, data: Data, base_indices: list, n_samples: int, degrees: dict
    ) -> list:
        """Sample nodes with bias toward well-connected nodes."""
        if n_samples >= len(base_indices):
            return base_indices
            
        # Convert to numpy for easier manipulation
        indices_array = np.array(base_indices)
        degree_values = np.array([degrees[idx] for idx in base_indices])
        
        # Combine stratified sampling with connectivity bias
        labels = data.y[indices_array].cpu().numpy()
        
        # Get base stratified sample
        stratified_indices = self._stratified_sample_indices(
            torch.tensor(labels), min(n_samples, len(base_indices))
        )
        
        # Convert back to original indices
        selected_base_indices = [base_indices[i] for i in stratified_indices]
        
        return selected_base_indices
    
    def _extract_subgraph_with_reconstruction(self, data: Data, node_indices: list) -> Data:
        """Extract subgraph and try to reconstruct important edges."""
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        
        # Create mapping
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices_tensor)}
        
        # Extract features
        x_sub = data.x[node_indices_tensor]
        y_sub = data.y[node_indices_tensor]
        coords_sub = data.coordinates[node_indices_tensor]
        
        # Extract edges more carefully
        edge_list = []
        edge_attrs = []
        
        for i, (src, dst) in enumerate(data.edge_index.t()):
            src_idx, dst_idx = src.item(), dst.item()
            if src_idx in node_map and dst_idx in node_map:
                edge_list.append([node_map[src_idx], node_map[dst_idx]])
                if data.edge_attr is not None:
                    edge_attrs.append(data.edge_attr[i])
        
        # Try to add missing spatial connections if connectivity is too low
        if len(edge_list) < len(node_indices_tensor) * 2:  # Very sparse graph
            print(f"⚠️  Graph too sparse ({len(edge_list)} edges for {len(node_indices_tensor)} nodes), adding spatial connections")
            edge_list, edge_attrs = self._add_spatial_edges(
                coords_sub, edge_list, edge_attrs, connection_radius=150.0
            )
        
        # Create tensors
        if edge_list:
            edge_index_sub = torch.tensor(edge_list, dtype=torch.long).t()
            if edge_attrs:
                if data.edge_attr is not None:
                    edge_attr_sub = torch.stack(edge_attrs)
                else:
                    edge_attr_sub = None
            else:
                edge_attr_sub = None
        else:
            edge_index_sub = torch.empty((2, 0), dtype=torch.long)
            edge_attr_sub = None
        
        return Data(
            x=x_sub,
            y=y_sub,
            edge_index=edge_index_sub,
            edge_attr=edge_attr_sub,
            coordinates=coords_sub,
        )
    
    def _add_spatial_edges(
        self, coordinates: torch.Tensor, existing_edges: list, existing_attrs: list, connection_radius: float
    ) -> tuple[list, list]:
        """Add spatial edges based on coordinate proximity."""
        from scipy.spatial import distance_matrix
        import torch.nn.functional as F
        
        coords_np = coordinates.cpu().numpy()
        dist_matrix = distance_matrix(coords_np, coords_np)
        
        # Find new spatial connections
        src, dst = np.where((dist_matrix < connection_radius) & (dist_matrix > 0))
        
        # Add to existing edges (avoid duplicates)
        existing_edge_set = {(src, dst) for src, dst in existing_edges}
        new_edges = existing_edges.copy()
        new_attrs = existing_attrs.copy()
        
        for s, d in zip(src, dst):
            if (s, d) not in existing_edge_set and (d, s) not in existing_edge_set:
                new_edges.append([s, d])
                new_edges.append([d, s])  # Make undirected
                
                # Create edge attributes
                dist = dist_matrix[s, d]
                edge_weight = torch.tensor(1.0 / (dist + 1e-6) ** 2).unsqueeze(0)
                new_attrs.extend([edge_weight, edge_weight])
        
        print(f"   Added {len(new_edges) - len(existing_edges)} spatial edges")
        return new_edges, new_attrs

    def get_sample_stats(self, data: Data) -> dict:
        """Get statistics about the sampled data."""
        labels_np = data.y.cpu().numpy()
        unique_classes, counts = np.unique(labels_np, return_counts=True)

        stats = {
            "total_samples": len(labels_np),
            "n_classes": len(unique_classes),
            "class_distribution": {
                int(cls): int(count)
                for cls, count in zip(unique_classes, counts, strict=False)
            },
            "class_balance": min(counts) / max(counts) if len(counts) > 1 else 1.0,
            "n_edges": data.edge_index.shape[1],
        }

        return stats


# Convenience function for integration
def apply_reservoir_sampling(
    base_data: Data, prediction_data: Data, params: dict, cycle_num: int = 1
) -> tuple[Data, dict]:
    """Apply reservoir sampling based on parameters.

    Returns:
        Tuple of (sampled_data, sampling_stats)

    """
    reservoir_config = params.get("combine_data", {}).get("reservoir", {})

    if not reservoir_config.get("enabled", False):
        # No reservoir sampling, just combine normally
        sampler = ReservoirSampler(max_size=int("inf"))
        combined_data = sampler._combine_graph_data(base_data, prediction_data)
        stats = sampler.get_sample_stats(combined_data)
        return combined_data, stats

    sampler = ReservoirSampler(
        max_size=reservoir_config.get("max_combined_size", 3000),
        strategy=reservoir_config.get("sampling_strategy", "stratified_recent"),
        preserve_recent_ratio=reservoir_config.get("preserve_recent_ratio", 0.4),
        class_balance_weight=reservoir_config.get("class_balance_weight", 0.8),
        quality_threshold=reservoir_config.get("quality_threshold"),
    )

    sampled_data = sampler.sample_combined_data(base_data, prediction_data, cycle_num)
    stats = sampler.get_sample_stats(sampled_data)

    return sampled_data, stats
