import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class ClusterPooling(nn.Module):
    def __init__(self, num_clusters, node_embedding_dim, temperature=1.0):
        super(ClusterPooling, self).__init__()
        
        self.num_clusters = num_clusters
        self.node_embedding_dim = node_embedding_dim
        self.temperature = temperature
        
        self.E = nn.Parameter(torch.randn(num_clusters, node_embedding_dim))
        nn.init.xavier_normal_(self.E)
        
    def forward(self, node_embeddings, data):
        similarities = torch.matmul(node_embeddings, self.E.t())  # [total_nodes_in_batch, num_clusters]
        
        P = F.softmax(similarities / self.temperature, dim=-1)  # [total_nodes_in_batch, num_clusters]
        
        dense_nodes, node_mask = to_dense_batch(node_embeddings, data.batch)
        
        dense_P, _ = to_dense_batch(P, data.batch)
        
        cluster_embeddings = torch.bmm(dense_P.transpose(1, 2), dense_nodes)
        
        return cluster_embeddings, P

class SecondOrderPooling(nn.Module):
    def __init__(self, node_embedding_dim, sopol_output_dim):
        super(SecondOrderPooling, self).__init__()
        
        self.node_embedding_dim = node_embedding_dim
        self.sopol_output_dim = sopol_output_dim
        
        # Linear mapping layer
        self.linear_map = nn.Linear(node_embedding_dim, sopol_output_dim)
        
    def forward(self, cluster_embeddings):
        mapped_clusters = self.linear_map(cluster_embeddings)
        
        sopol_matrix = torch.bmm(mapped_clusters.transpose(1, 2), mapped_clusters)
        
        batch_size = sopol_matrix.size(0)
        graph_embeddings = sopol_matrix.view(batch_size, -1)
        
        return graph_embeddings

class HierarchicalPooling(nn.Module):
    def __init__(self, num_clusters, node_embedding_dim, sopol_output_dim, num_classes, clustering_temperature=1.0):
        super(HierarchicalPooling, self).__init__()
        
        self.num_clusters = num_clusters
        self.node_embedding_dim = node_embedding_dim
        self.sopol_output_dim = sopol_output_dim
        self.num_classes = num_classes
        self.clustering_temperature = clustering_temperature
        
        self.cluster_pooling = ClusterPooling(num_clusters, node_embedding_dim, clustering_temperature)
        
        self.second_order_pooling = SecondOrderPooling(node_embedding_dim, sopol_output_dim)
        
        self.cluster_norm = nn.LayerNorm(node_embedding_dim)
        
        self.output_dim = sopol_output_dim * sopol_output_dim
        
    def forward(self, node_embeddings, data):
        cluster_embeddings, P = self.cluster_pooling(node_embeddings, data)
        
        batch_size, num_clusters, embedding_dim = cluster_embeddings.shape
        cluster_embeddings_flat = cluster_embeddings.view(-1, embedding_dim)
        cluster_embeddings_normalized = self.cluster_norm(cluster_embeddings_flat)
        cluster_embeddings = cluster_embeddings_normalized.view(batch_size, num_clusters, embedding_dim)
        
        graph_embeddings = self.second_order_pooling(cluster_embeddings)
        
        return graph_embeddings, P
 