import torch
from torch_geometric.utils import to_dense_adj, dense_to_sparse, degree
from torch_scatter import scatter_sum, scatter_add

def calculate_differentiable_modularity_loss(data, P):
    edge_index = data.edge_index
    batch = data.batch
    num_nodes = data.num_nodes
    batch_size = batch.max().item() + 1
    num_clusters = P.size(1)
    
    node_degrees = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    
    total_degree_per_graph = scatter_sum(node_degrees, batch, dim=0)  # Shape: [batch_size]
    m_per_graph = total_degree_per_graph / 2.0  
    
    row, col = edge_index[0], edge_index[1]
    edge_contributions = torch.sum(P[row] * P[col], dim=1)  # Shape: [num_edges]
    first_term_per_graph = scatter_sum(edge_contributions, batch[row], dim=0)  # Shape: [batch_size]
    
    degree_weighted_P = node_degrees.unsqueeze(1) * P  # Shape: [num_nodes, num_clusters]
    
    batch_expanded = batch.unsqueeze(1).expand(-1, num_clusters)  # Shape: [num_nodes, num_clusters]
    cluster_idx = torch.arange(num_clusters, device=P.device).unsqueeze(0).expand(num_nodes, -1)  # Shape: [num_nodes, num_clusters]
    
    combined_idx = batch_expanded * num_clusters + cluster_idx  # Shape: [num_nodes, num_clusters]
    
    degree_weighted_P_flat = degree_weighted_P.flatten()  # Shape: [num_nodes * num_clusters]
    combined_idx_flat = combined_idx.flatten()  # Shape: [num_nodes * num_clusters]
    
    dT_P_flat = scatter_sum(degree_weighted_P_flat, combined_idx_flat, 
                           dim=0, dim_size=batch_size * num_clusters)  # Shape: [batch_size * num_clusters]
    
    dT_P_per_graph = dT_P_flat.view(batch_size, num_clusters)  # Shape: [batch_size, num_clusters]
    
    second_term_per_graph = torch.sum(dT_P_per_graph ** 2, dim=1)  # Shape: [batch_size]
    
    valid_graphs = m_per_graph > 0
    modularity_scores = torch.zeros_like(m_per_graph)
    
    if valid_graphs.any():
        valid_m = m_per_graph[valid_graphs]
        valid_first = first_term_per_graph[valid_graphs]
        valid_second = second_term_per_graph[valid_graphs]
        
        modularity_scores[valid_graphs] = (
            valid_first / (2.0 * valid_m) - 
            valid_second / (4.0 * valid_m * valid_m)
        )
    
    if valid_graphs.any():
        modularity_batch_score = modularity_scores[valid_graphs].mean()
    else:
        modularity_batch_score = torch.tensor(0.0, device=P.device)
    
    modularity_loss = -modularity_batch_score
    
    return modularity_loss 