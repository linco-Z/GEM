import torch
import numpy as np
import math
from scipy.io import loadmat
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, degree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from utils.graph_metrics import add_betweenness_centrality_to_data_list, add_node_metrics_to_data_list

# Dataset configurations
DATASET_CONFIGS = {
    # Brain network datasets
    'ABIDE': {
        'type': 'brain',
        'data_path': '/opt/data/private/00_data/ABIDE.mat',
        'num_classes': 2,
        'num_features': 116,
        'sparsity_threshold': 0.2
    },
    'SRPBS': {
        'type': 'brain',
        'data_path': '/opt/data/private/00_data/SRPBS.mat',
        'num_classes': 2,
        'num_features': 116,
        'sparsity_threshold': 0.2
    },
    # Molecular datasets
    'MUTAG': {
        'type': 'molecular',
        'root': '/opt/data/private/00_data/TUDataset/Bio',
        'num_classes': 2,
        'num_features': 7,  # Will be auto-detected
    },
    'PROTEINS': {
        'type': 'molecular',
        'root': '/opt/data/private/00_data/TUDataset/Bio',
        'num_classes': 2,
        'num_features': 3,  # Will be auto-detected
    },
    'NCI1': {
        'type': 'molecular',
        'root': '/opt/data/private/00_data/TUDataset/Bio',
        'num_classes': 2,
        'num_features': 37,  # Will be auto-detected
    },
    # Social network datasets
    'COLLAB': {
        'type': 'social',
        'root': '/opt/data/private/00_data/TUDataset/Social',
        'num_classes': 3,
        'num_features': 1,  # Will be auto-detected (degree features)
    },
    'REDDIT-BINARY': {
        'type': 'social',
        'root': '/opt/data/private/00_data/TUDataset/Social',
        'num_classes': 2,
        'num_features': 1,  # Will be auto-detected (degree features)
}

def load_dataset_tu(dataset_name, root_path):
    print(f'Loading TU dataset: {dataset_name}')
    dataset = TUDataset(root=root_path, name=dataset_name)
    data_list = [data for data in dataset]
    
    if data_list[0].x is None:
        print(f"No node features found for {dataset_name}, using degree as temporary features")
        for data in data_list:
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
            data.x = deg.view(-1, 1)
    
    print(f"Loaded {len(data_list)} graphs from {dataset_name}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {data_list[0].x.size(1)}")
    
    return data_list


def load_dataset_brain(graph, sparsity_threshold=0.2):
    print('Loading brain network data...')
    labels = graph["label"]
    num_graphs = len(labels)
    data_list = []
    for i in range(num_graphs):
        node_features = torch.FloatTensor(graph["graph_struct"][0][i]["ROI"])
        FC = node_features
        
        tepk = FC.reshape(-1, 1)
        tepk, _ = torch.sort(abs(tepk), dim=0, descending=True)
        mk = tepk[int(math.pow(FC.shape[0], 2) * sparsity_threshold)]  
        adj = torch.Tensor(np.where(FC > mk, 1, 0))

        label = labels[i, 0]
        data_example = Data(
            x=node_features, 
            edge_index=dense_to_sparse(adj)[0],  
            y=torch.tensor(label, dtype=torch.long),
            edge_attr=None
        )
        data_list.append(data_example)

    print(f"Loaded {len(data_list)} brain network graphs")
    return data_list


def load_dataset(data_config):
    dataset_name = data_config['dataset_name']
    dataset_info = DATASET_CONFIGS[dataset_name]
    
    node_metric_type = data_config.get('node_metric_type', 'betweenness_centrality')
    print(f"Using node metric type: {node_metric_type}")
    
    if dataset_info['type'] == 'brain':
        data_path = dataset_info['data_path']
        sparsity_threshold = data_config.get('sparsity_threshold', dataset_info['sparsity_threshold'])
        graph = loadmat(data_path)
        data_list = load_dataset_brain(graph, sparsity_threshold)
        
        print(f"Adding {node_metric_type} for node ordering to {dataset_name} dataset...")
        data_list = add_node_metrics_to_data_list(
            data_list, 
            dataset_name,
            metric_type=node_metric_type,
            use_for_ordering=True,
            use_as_features=False,
            cache_root="cache"
        )
    else:
        root_path = dataset_info['root']
        data_list = load_dataset_tu(dataset_name, root_path)
        
        if dataset_info['type'] == 'social':
            print(f"Adding {node_metric_type} to social network dataset {dataset_name}...")
            data_list = add_node_metrics_to_data_list(
                data_list, 
                dataset_name,
                metric_type=node_metric_type,
                use_for_ordering=True,
                use_as_features=True,
                cache_root="cache"
            )
        else:
            print(f"Adding {node_metric_type} for node ordering to molecular dataset {dataset_name}...")
            data_list = add_node_metrics_to_data_list(
                data_list, 
                dataset_name,
                metric_type=node_metric_type,
                use_for_ordering=True,
                use_as_features=False,
                cache_root="cache"
            )
    
    data_config['num_features'] = data_list[0].x.size(1)
    data_config['num_classes'] = len(set([data.y.item() for data in data_list]))
    
    print(f"Final dataset info:")
    print(f"Number of features: {data_config['num_features']}")
    print(f"Number of classes: {data_config['num_classes']}")
    
    return data_list

def split_dataset_fixed(dataset, data_config, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset_seed = data_config.get('dataset_seed', 42)
    
    current_np_state = np.random.get_state()
    
    np.random.seed(dataset_seed)
    
    labels = [data.y.item() for data in dataset]
    indices = list(range(len(dataset)))
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=dataset_seed
    )
    
    train_val_labels = [labels[i] for i in train_val_idx]
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)  # adjust val ratio in remaining data
    
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted, stratify=train_val_labels, random_state=dataset_seed
    )
    
    np.random.set_state(current_np_state)
    
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    
    print(f"Dataset split with dataset_seed {dataset_seed}:")
    print(f"Train: {len(train_data)} ({len(train_data)/len(dataset):.1%})")
    print(f"Val: {len(val_data)} ({len(val_data)/len(dataset):.1%})")
    print(f"Test: {len(test_data)} ({len(test_data)/len(dataset):.1%})")
    
    return train_data, val_data, test_data

def get_fixed_dataloaders(dataset, training_config, data_config, mode='hyperparameter_tuning'):
    batch_size = training_config['batch_size']
    
    train_data, val_data, test_data = split_dataset_fixed(dataset, data_config=data_config)
    
    main_seed = training_config.get('seed', 9)
    generator = torch.Generator()
    generator.manual_seed(main_seed)
    
    if mode == 'hyperparameter_tuning':
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    
    elif mode == 'final_training':
        train_val_data = train_data + val_data
        train_val_loader = DataLoader(train_val_data, batch_size=batch_size, shuffle=True, generator=generator)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_val_loader, test_loader
    
    elif mode == 'fixed_811':
        # fixed 8:1:1 split: return separate train, val, and test loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    else:
        raise ValueError(f"Unknown mode: {mode}")