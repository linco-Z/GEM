import torch
import torch.nn as nn

class MLP_subgraph(torch.nn.Module):
 
    def __init__(self, device, node_feature_dim):
        super(MLP_subgraph, self).__init__()

        self.device = device
        self.node_feature_dim = node_feature_dim
        self.reg_coefs = (0.001, 0.001)
        
        initial_dim = 2 * node_feature_dim
        self.explainer_model = nn.Sequential(
            nn.Linear(initial_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)
        
        self._initialized = False
        self._current_input_dim = initial_dim

    def _update_model_if_needed(self, total_feature_dim):
        if total_feature_dim != self._current_input_dim:
            old_hidden_weight = self.explainer_model[2].weight.data.clone()
            old_hidden_bias = self.explainer_model[2].bias.data.clone()
            
            self.explainer_model = nn.Sequential(
                nn.Linear(total_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            ).to(self.device)
            
            self.explainer_model[2].weight.data = old_hidden_weight
            self.explainer_model[2].bias.data = old_hidden_bias
            
            self._current_input_dim = total_feature_dim
        
        self._initialized = True

    def _create_explainer_input(self, pair, embeds, edge_attr=None):
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]

        input_expl = torch.cat([row_embeds, col_embeds], 1)
        
        if edge_attr is not None:
            input_expl = torch.cat([input_expl, edge_attr], 1)
        
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        sampling_weights = torch.clamp(sampling_weights, -10, 10)
        
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            # Use torch.rand with proper device to ensure reproducibility with manual_seed
            eps = (bias - (1-bias)) * torch.rand(sampling_weights.size(), device=self.device) + (1-bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        
        return graph

    def forward(self, data):
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
        
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr.to(self.device)
        
        input_expl = self._create_explainer_input(edge_index.to(self.device), x, edge_attr)
        
        if not self._initialized:
            total_feature_dim = input_expl.size(1)
            self._update_model_if_needed(total_feature_dim)
        
        sampling_weights = self.explainer_model(input_expl)

        edge_mask = self._sample_graph(sampling_weights, training=self.training).squeeze()
        return edge_mask