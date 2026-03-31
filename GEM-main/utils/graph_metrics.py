import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx, degree
from torch_geometric.data import Data
import warnings
import pickle
import hashlib
import os
from pathlib import Path
warnings.filterwarnings('ignore')


def compute_betweenness_centrality(edge_index, num_nodes):
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), 
                   to_undirected=True)
    betweenness = nx.betweenness_centrality(G)
    scores = torch.zeros(num_nodes)
    for node, score in betweenness.items():
        scores[node] = score
    return scores


def compute_node_degree(edge_index, num_nodes):
    row, col = edge_index
    deg = degree(row, num_nodes, dtype=torch.float)
    normalized_deg = deg / (num_nodes - 1) if num_nodes > 1 else deg
    return normalized_deg


def compute_node_metrics(edge_index, num_nodes, metric_type="betweenness_centrality"):
    if metric_type == "betweenness_centrality":
        return compute_betweenness_centrality(edge_index, num_nodes)
    elif metric_type == "node_degree":
        return compute_node_degree(edge_index, num_nodes)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}. Supported types: 'betweenness_centrality', 'node_degree'")


def _get_graph_hash(edge_index, num_nodes):
    edge_list = edge_index.t().numpy()
    edge_list_sorted = np.sort(edge_list, axis=1)
    edge_list_sorted = edge_list_sorted[np.lexsort((edge_list_sorted[:, 1], edge_list_sorted[:, 0]))]
    
    graph_str = f"{num_nodes}_{edge_list_sorted.tobytes()}"
    return hashlib.md5(graph_str.encode()).hexdigest()


def _create_cache_directory(cache_root="cache"):
    cache_dir = Path(cache_root)
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def _load_dataset_cache(dataset_name, metric_type, cache_root="cache"):
    cache_dir = _create_cache_directory(cache_root)
    cache_file = cache_dir / f"{dataset_name}_{metric_type}_cache.pkl"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            print(f"✓ Loaded {metric_type} cache for {dataset_name} ({len(cache_data)} graphs)")
            return cache_data
        except Exception as e:
            print(f"Warning: Failed to load cache for {dataset_name}: {e}")
            return {}
    return {}


def _save_dataset_cache(dataset_name, metric_type, cache_data, cache_root="cache"):
    cache_dir = _create_cache_directory(cache_root)
    cache_file = cache_dir / f"{dataset_name}_{metric_type}_cache.pkl"
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved {metric_type} cache for {dataset_name} ({len(cache_data)} graphs)")
    except Exception as e:
        print(f"Warning: Failed to save cache for {dataset_name}: {e}")


def add_node_metrics_to_data_list(data_list, dataset_name, metric_type="betweenness_centrality", use_for_ordering=True, use_as_features=False, cache_root="cache"):
    print(f"Computing {metric_type} for {dataset_name} dataset ({len(data_list)} graphs)")
    
    cache_data = _load_dataset_cache(dataset_name, metric_type, cache_root)
    
    cache_hits = 0
    cache_misses = 0
    new_computations = {}
    
    for i, data in enumerate(data_list):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processing graph {i+1}/{len(data_list)} (cache hits: {cache_hits}, misses: {cache_misses})")
        
        graph_hash = _get_graph_hash(data.edge_index, data.x.size(0))
        
        if graph_hash in cache_data:
            metric_scores = cache_data[graph_hash]
            cache_hits += 1
        else:
            metric_scores = compute_node_metrics(data.edge_index, data.x.size(0), metric_type)
            new_computations[graph_hash] = metric_scores
            cache_misses += 1
        
        if use_for_ordering:
            data.node_ordering_scores = metric_scores
        
        if use_as_features:
            metric_features = metric_scores.unsqueeze(1)
            
            if hasattr(data, 'x') and data.x is not None and data.x.size(1) > 1:
                data.x = torch.cat([data.x, metric_features], dim=1)
            else:
                data.x = metric_features
    
    if new_computations:
        cache_data.update(new_computations)
        _save_dataset_cache(dataset_name, metric_type, cache_data, cache_root)
    
    print(f"✓ Added {metric_type} to {len(data_list)} graphs")
    print(f"  Cache statistics: {cache_hits} hits, {cache_misses} misses ({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")
    if new_computations:
        print(f"  Computed and cached {len(new_computations)} new graphs")
    
    if use_for_ordering:
        print("  - Used for node ordering (Mamba sequencing)")
    if use_as_features:
        print("  - Used as node features")
    
    return data_list


def add_betweenness_centrality_to_data_list(data_list, dataset_name, use_for_ordering=True, use_as_features=False, cache_root="cache"):
    return add_node_metrics_to_data_list(
        data_list, dataset_name, 
        metric_type="betweenness_centrality",
        use_for_ordering=use_for_ordering, 
        use_as_features=use_as_features, 
        cache_root=cache_root
    )


def add_node_degree_to_data_list(data_list, dataset_name, use_for_ordering=True, use_as_features=False, cache_root="cache"):
    return add_node_metrics_to_data_list(
        data_list, dataset_name, 
        metric_type="node_degree",
        use_for_ordering=use_for_ordering, 
        use_as_features=use_as_features, 
        cache_root=cache_root
    ) 