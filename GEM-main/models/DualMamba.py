import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch

class DualMamba(nn.Module):
    def __init__(self, cfg_data, cfg_model, input_dim):
        super(DualMamba, self).__init__()
        
        self.cfg_data = cfg_data
        self.cfg_model = cfg_model
        self.input_dim = input_dim
        self.hidden_dim = cfg_model['hidden_dim']
        
        self.d_state = cfg_model.get('mamba_d_state', 16)
        self.d_conv = cfg_model.get('mamba_d_conv', 4)
        self.expand = cfg_model.get('mamba_expand', 2)
        
        self.forward_mamba = Mamba(
            d_model=self.hidden_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        
        self.backward_mamba = Mamba(
            d_model=self.hidden_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        
    
    def _batch_aware_sort(self, node_ordering_scores, batch):
        if node_ordering_scores is None:
            node_ordering_scores = torch.rand(len(batch), device=batch.device)
        
        max_score = node_ordering_scores.max() + 1
        num_nodes = len(batch)
        
        sorting_keys = batch * (max_score + 1) - node_ordering_scores
        
        sorted_indices = torch.argsort(sorting_keys)
        
        unsort_indices = torch.argsort(sorted_indices)
        
        return sorted_indices, unsort_indices
    
    def forward(self, node_features, batch, node_ordering_scores=None):
        device = node_features.device
        
        sorted_indices, unsort_indices = self._batch_aware_sort(node_ordering_scores, batch)
        
        sorted_features = node_features[sorted_indices]  # [num_nodes, hidden_dim]
        sorted_batch = batch[sorted_indices]  # [num_nodes]
        
        dense_features, mask = to_dense_batch(sorted_features, sorted_batch)
        batch_size, max_seq_len, hidden_dim = dense_features.shape
        
        forward_out = self.forward_mamba(dense_features)  # [batch_size, max_seq_len, hidden_dim]
        
        reversed_features = torch.flip(dense_features, dims=[1])  # [batch_size, max_seq_len, hidden_dim]
        backward_out = self.backward_mamba(reversed_features)  # [batch_size, max_seq_len, hidden_dim]
        backward_out = torch.flip(backward_out, dims=[1])
        
        combined_out = forward_out + backward_out  # [batch_size, max_seq_len, hidden_dim]
        
        valid_mask = mask.view(-1)  # [batch_size * max_seq_len]
        combined_flat = combined_out.view(-1, hidden_dim)  # [batch_size * max_seq_len, hidden_dim]
        
        valid_embeddings = combined_flat[valid_mask]  # [num_nodes, hidden_dim]
        
        output_embeddings = valid_embeddings[unsort_indices]  # [num_nodes, hidden_dim]
        
        return output_embeddings

class DualMambaMulti(nn.Module):
    def __init__(self, cfg_data, cfg_model, input_dim):
        super(DualMambaMulti, self).__init__()
        
        self.cfg_data = cfg_data
        self.cfg_model = cfg_model
        self.input_dim = input_dim
        self.hidden_dim = cfg_model['hidden_dim']
        
        self.d_state = cfg_model.get('mamba_d_state', 16)
        self.d_conv = cfg_model.get('mamba_d_conv', 4)
        self.expand = cfg_model.get('mamba_expand', 2)
        
        self.input_projection = nn.Linear(input_dim, self.hidden_dim)
        
        self.forward_mamba = Mamba(
            d_model=self.hidden_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        
        self.backward_mamba = Mamba(
            d_model=self.hidden_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
        )
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        self.dropout = nn.Dropout(cfg_model.get('dropout_rate', 0.1))
    
    def _batch_aware_sort(self, node_ordering_scores, batch):
        if node_ordering_scores is None:
            node_ordering_scores = torch.rand(len(batch), device=batch.device)
        
        max_score = node_ordering_scores.max() + 1
        
        sorting_keys = batch * (max_score + 1) - node_ordering_scores
        
        sorted_indices = torch.argsort(sorting_keys)
        
        unsort_indices = torch.argsort(sorted_indices)
        
        return sorted_indices, unsort_indices
    
    def forward(self, node_features, batch, node_ordering_scores=None):
        device = node_features.device
        
        projected_features = self.input_projection(node_features)
        projected_features = self.layer_norm(projected_features)
        
        sorted_indices, unsort_indices = self._batch_aware_sort(node_ordering_scores, batch)
        
        sorted_features = projected_features[sorted_indices]
        sorted_batch = batch[sorted_indices] 
        
        dense_features, mask = to_dense_batch(sorted_features, sorted_batch)
        batch_size, max_seq_len, hidden_dim = dense_features.shape
        
        forward_out = self.forward_mamba(dense_features)  # [batch_size, max_seq_len, hidden_dim]
        
        reversed_features = torch.flip(dense_features, dims=[1])
        backward_out = self.backward_mamba(reversed_features)
        backward_out = torch.flip(backward_out, dims=[1])
        
        combined_out = forward_out + backward_out  # [batch_size, max_seq_len, hidden_dim]
        
        combined_out = self.dropout(combined_out)
        
        valid_mask = mask.view(-1)  # [batch_size * max_seq_len]
        combined_flat = combined_out.view(-1, hidden_dim)
        valid_embeddings = combined_flat[valid_mask]
        
        output_embeddings = valid_embeddings[unsort_indices]
        
        return output_embeddings
