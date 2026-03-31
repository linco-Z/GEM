import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter_add
from torch.nn.init import xavier_uniform_, zeros_

def gcn_norm(edge_index, edge_weight, num_nodes=None, do_add_self_loops=True):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if do_add_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)

    row, col = edge_index
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return edge_index, edge_weight

class IGMPConv(MessagePassing):
    def __init__(self, in_channels, out_channels, add_self_loops=True, normalize=True, bias=True):
        super(IGMPConv, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        
        self.lin = nn.Linear(out_channels * 2 + 1, out_channels)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        xavier_uniform_(self.weight)
        xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            zeros_(self.bias)
        zeros_(self.lin.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        original_edge_weight = edge_weight.clone()
        
        if self.normalize:
            edge_index_norm, edge_weight_norm = gcn_norm(
                edge_index, edge_weight, x.size(0), self.add_self_loops
            )
            
            if self.add_self_loops:
                if edge_index_norm.size(1) > edge_index.size(1):
                    num_added_loops = edge_index_norm.size(1) - edge_index.size(1)
                    self_loop_weights = torch.ones(num_added_loops, device=edge_weight.device)
                    original_edge_weight = torch.cat([original_edge_weight, self_loop_weights], dim=0)
                
                norm_factors = torch.where(
                    original_edge_weight.abs() > 1e-8, 
                    edge_weight_norm / original_edge_weight,
                    edge_weight_norm
                )
            else:
                norm_factors = torch.where(
                    original_edge_weight.abs() > 1e-8,
                    edge_weight_norm / original_edge_weight,
                    edge_weight_norm
                )
            
            edge_index = edge_index_norm
            
        else:
            if self.add_self_loops:
                edge_index, original_edge_weight = add_self_loops(
                    edge_index, original_edge_weight, num_nodes=x.size(0)
                )
            norm_factors = torch.ones_like(original_edge_weight)
        
        x = x @ self.weight
        
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        
        msg = self.message_content(x_i, x_j, original_edge_weight)
        
        scaled_msg = msg * norm_factors.view(-1, 1)
        
        out = scatter_add(scaled_msg, col, dim=0, dim_size=x.size(0))
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message_content(self, x_i, x_j, edge_weight):
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.view(-1, 1)
        
        msg = torch.cat([x_i, x_j, edge_weight], dim=1)
        return self.lin(msg)
    
    def message(self, x_i, x_j, edge_weight):
        return self.message_content(x_i, x_j, edge_weight)
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter_add(inputs, index, dim=0, dim_size=dim_size)

class IGMP(nn.Module):
    def __init__(self, cfg_data, cfg_model, input_dim=None):
        super(IGMP, self).__init__()
        
        self.in_channels = input_dim if input_dim is not None else cfg_data['num_features']
        self.hidden_dim = cfg_model['hidden_dim']
        
        self.conv = IGMPConv(self.in_channels, self.hidden_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        node_emb = self.conv(x, edge_index, edge_weight)
        
        return node_emb

class IGMPMulti(nn.Module):
    def __init__(self, cfg_data, cfg_model, input_dim=None):
        super(IGMPMulti, self).__init__()
        
        self.in_channels = input_dim if input_dim is not None else cfg_data['num_features']
        self.num_layers = cfg_model.get('num_igmp_layers', 2)
        self.hidden_dim = cfg_model['hidden_dim']
        self.dropout_rate = cfg_model.get('dropout_rate', 0.1)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.convs.append(IGMPConv(self.in_channels, self.hidden_dim))
        self.batch_norms.append(nn.LayerNorm(self.hidden_dim))
        
        for _ in range(self.num_layers - 1):
            self.convs.append(IGMPConv(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(nn.LayerNorm(self.hidden_dim))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = self.dropout(x)
            x = self.relu(x)
        node_emb = x

        return node_emb

