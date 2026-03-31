import torch.nn as nn
import torch.nn.functional as F
from models.IGMP import IGMP
from models.DualMamba import DualMamba
from models.HierarchicalPooling import HierarchicalPooling

class IGConv(nn.Module):
    def __init__(self, cfg_data, cfg_model, input_dim):
        super(IGConv, self).__init__()
        self.hidden_dim = cfg_model['hidden_dim']
        self.input_dim = input_dim
        
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        self.igmp = IGMP(cfg_data, cfg_model, self.hidden_dim)
        
        self.mamba = DualMamba(cfg_data, cfg_model, self.hidden_dim)
        
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg_model.get('dropout_rate', 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    def forward(self, x, edge_index, batch, node_ordering_scores=None, edge_weight=None):
        x_projected = self.input_projection(x)
        
        igmp_embedding = self.igmp(x_projected, edge_index, edge_weight)
        
        mamba_embedding = self.mamba(x_projected, batch, node_ordering_scores)
        
        fused_embedding = igmp_embedding + mamba_embedding
        fused_embedding = self.fusion_ffn(fused_embedding)
        
        return fused_embedding

class IGModel(nn.Module):
    def __init__(self, config):
        super(IGModel, self).__init__()
        
        self.cfg_data = config['DATA_CONFIG']
        self.cfg_model = config['MODEL_CONFIG']
        self.hidden_dim = self.cfg_model['hidden_dim']
        self.num_classes = self.cfg_data['num_classes']
        self.input_dim = self.cfg_data['num_features']
        
        self.num_ig_layers = self.cfg_model.get('num_ig_layers', 3)
        
        self.ig_convs = nn.ModuleList([
            IGConv(self.cfg_data, self.cfg_model, input_dim=self.input_dim if i == 0 else self.hidden_dim)
            for i in range(self.num_ig_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim)
            for _ in range(self.num_ig_layers)
        ])
        
        self.pool = HierarchicalPooling(
            num_clusters=self.cfg_model.get('num_clusters', 8),
            node_embedding_dim=self.hidden_dim,
            sopol_output_dim=self.cfg_model.get('sopol_output_dim', 32),
            num_classes=self.num_classes,
            clustering_temperature=self.cfg_model.get('clustering_temperature', 1.0)
        )
        
        num_mlp_layers = self.cfg_model.get('num_mlp_layers', 2)
        mlp_hidden_dim = self.cfg_model.get('mlp_hidden_dim', 128)
        dropout_rate = self.cfg_model.get('dropout_rate', 0.1)
        
        if num_mlp_layers == 1:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(self.pool.output_dim, self.num_classes)
            )
        else:
            layers = []
            input_dim = self.pool.output_dim
            for i in range(num_mlp_layers - 1):
                layers.extend([
                    nn.Linear(input_dim, mlp_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_dim = mlp_hidden_dim
            layers.append(nn.Linear(input_dim, self.num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, batch):
        x = batch.x
        node_ordering_scores = batch.node_ordering_scores if hasattr(batch, 'node_ordering_scores') else None
        
        edge_weight = batch.edge_weight if hasattr(batch, 'edge_weight') else None
        
        for i, (ig_conv, layer_norm) in enumerate(zip(self.ig_convs, self.layer_norms)):
            if i == 0:
                x = ig_conv(x, batch.edge_index, batch.batch, node_ordering_scores, edge_weight)
                x = layer_norm(x)
            else:
                residual = x
                x = layer_norm(x)
                x_out = ig_conv(x, batch.edge_index, batch.batch, node_ordering_scores, edge_weight)
                x = residual + x_out
        
        graph_embedding, P = self.pool(x, batch)
        
        logits = self.classifier(graph_embedding)
        
        return graph_embedding, logits, P
