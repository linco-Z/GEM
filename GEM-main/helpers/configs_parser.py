import os
import yaml
from typing import Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, WandB features will be disabled")

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    default_config = {
        "DATA_CONFIG": {
            "sparsity_threshold": 0.2,
            "node_ordering_metric": "degree",
            "cluster_selection_metric": "within_module_degree"
        },
        "MODEL_CONFIG": {
            "hidden_dim": 128,
            "num_ig_layers": 2,
            "num_igmp_layers": 2,
            "num_mamba_layers": 2,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "pooling_type": "mean",
            "use_residual": True,
            "dropout_rate": 0.5,
            "num_clusters": 16,
            "pool_hidden_dim": 64,
            "pool_output_dim": 32
        },
        "TRAINING_CONFIG": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "device": "cuda",
            "seed": 42,
            "optimizer_type": "Adam",
            "scheduler_type": "ReduceLROnPlateau",
            "scheduler_mode": "min",
            "scheduler_factor": 0.5,
            "scheduler_patience": 10,
            "scheduler_min_lr": 1e-6,
            "scheduler_monitor_metric": "val_loss",
            "mi_weight": 0.01,
            "gt_weight": 0.001,

            "early_stopping": True,
            "early_stopping_patience": 10,
            "early_stopping_delta": 0.001,
            "sigma": 30
        },
        "WANDB_CONFIG": {
            "use_wandb": False,
            "project_name": "IGModel",
            "entity_name": "linco-project",
            "run_name": "IGModel_CV",
            "wandb_log_model_freq": 100
        }
    }
    
    def merge_configs(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_configs(base[key], value)
            else:
                base[key] = value
        return base
    
    validated_config = merge_configs(default_config, config)
    
    assert validated_config["DATA_CONFIG"]["num_classes"] > 0, "num_classes must be positive"
    assert validated_config["DATA_CONFIG"]["num_features"] > 0, "num_features must be positive"
    assert validated_config["MODEL_CONFIG"]["hidden_dim"] > 0, "hidden_dim must be positive"
    assert validated_config["TRAINING_CONFIG"]["learning_rate"] > 0, "learning_rate must be positive"
    
    return validated_config

def load_configs(yaml_path, cli_args=None):
    if not os.path.exists(yaml_path):
        print(f"warning: YAML config {yaml_path} not exist, use default configs.")
        config = {"MODEL_CONFIG": {}, "TRAINING_CONFIG": {}, "WANDB_CONFIG": {}, "DATA_CONFIG": {}}
    else:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            if config is None: 
                config = {"MODEL_CONFIG": {}, "TRAINING_CONFIG": {}, "WANDB_CONFIG": {}, "DATA_CONFIG": {}}
            config.setdefault("MODEL_CONFIG", {})
            config.setdefault("TRAINING_CONFIG", {})
            config.setdefault("WANDB_CONFIG", {})
            config.setdefault("DATA_CONFIG", {})

    def convert_yaml_none(config_dict):
        for section_name, section in config_dict.items():
            if isinstance(section, dict):
                for key, value in section.items():
                    if value == "None" or value == "null":
                        section[key] = None
                    elif isinstance(value, dict):
                        convert_yaml_none({key: value})
        return config_dict
    
    config = convert_yaml_none(config)
    
    if cli_args:
        for key, value in vars(cli_args).items():
            if value is not None:
                if key in config["TRAINING_CONFIG"]:
                    config["TRAINING_CONFIG"][key] = value
                elif key in config["WANDB_CONFIG"]:
                    config["WANDB_CONFIG"][key] = value
                elif key in config["MODEL_CONFIG"]: 
                    config["MODEL_CONFIG"][key] = value
                elif key in config["DATA_CONFIG"]:
                    config["DATA_CONFIG"][key] = value
                else:
                    config["TRAINING_CONFIG"][key] = value

    # 3. WandB
    if config.get("WANDB_CONFIG", {}).get("use_wandb", False):
        wandb.init(
            project=config["WANDB_CONFIG"].get("project_name", "default_project"),
            entity=config["WANDB_CONFIG"].get("entity_name", None),
            config=config,
            settings=wandb.Settings(start_method="thread")
        )
        
        wandb_dict = dict(wandb.config)
        final_config = {
            "MODEL_CONFIG": config["MODEL_CONFIG"].copy(),
            "TRAINING_CONFIG": config["TRAINING_CONFIG"].copy(),
            "WANDB_CONFIG": config["WANDB_CONFIG"].copy(),
            "DATA_CONFIG": config["DATA_CONFIG"].copy()
        }
        
        for key, value in wandb_dict.items():
            if key in config["MODEL_CONFIG"]:
                final_config["MODEL_CONFIG"][key] = value
            elif key in config["TRAINING_CONFIG"]:
                final_config["TRAINING_CONFIG"][key] = value
            elif key in config["WANDB_CONFIG"]:
                final_config["WANDB_CONFIG"][key] = value
            elif key in config["DATA_CONFIG"]:
                final_config["DATA_CONFIG"][key] = value
        
        print("WandB Initialized, configs updated from wandb.config")
        return validate_config(final_config)
    else:
        print("WandB deactivated")
        return validate_config(config)