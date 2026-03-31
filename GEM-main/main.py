import sys
import wandb 
import argparse
import numpy as np
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.IGModel import IGModel
from models.subgraph_generator import MLP_subgraph
from helpers.configs_parser import load_configs 
from helpers.logger import logger
from utils.data_loader import load_dataset, get_fixed_dataloaders, get_kfold_dataloaders
from utils.ib_loss import CS_QMI_normalized
from utils.modularity_loss import calculate_differentiable_modularity_loss

DATASET_CONFIGS = {
    'ABIDE': 'configs/ABIDE.yaml', 
    'SRPBS': 'configs/SRPBS.yaml',
    'MUTAG': 'configs/MUTAG.yaml',
    'PROTEINS': 'configs/PROTEINS.yaml',
    'NCI1': 'configs/NCI1.yaml',
    'COLLAB': 'configs/COLLAB.yaml',
    'REDDIT-BINARY': 'configs/REDDIT-BINARY.yaml',
}

def calc_performance_statistics(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, f1, mcc

def train_one_epoch(model, sg_model, dataloader, optimizer, device, epoch_num, training_config, use_wandb):
    model.train()
    sg_model.train()
    
    total_loss = 0.0
    total_classify_loss = 0.0
    total_mi_loss = 0.0
    total_modularity_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Get loss weights
    mi_weight = training_config.get('mi_weight', 0.01)
    lambda_mod = training_config.get('lambda_mod', 0.5) 
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num:3d} [Train]", 
                leave=False, file=sys.stdout, ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()

        original_graph_embeddings, _, _ = model(batch)
        
        batch_subgraph = batch.clone()
        edge_mask = sg_model(batch_subgraph)
        
        batch_subgraph.edge_weight = edge_mask  
        
        subgraph_embeddings, subgraph_logits, subgraph_P = model(batch_subgraph)
        
        classify_loss = F.cross_entropy(subgraph_logits, batch.y)
        
        if mi_weight > 0:
            sigma = training_config.get('sigma', None)
            mi_loss = CS_QMI_normalized(original_graph_embeddings, subgraph_embeddings, sigma)
        else:
            mi_loss = torch.tensor(0.0, device=batch.y.device)
            
        modularity_loss = calculate_differentiable_modularity_loss(batch_subgraph, subgraph_P)
        loss = classify_loss + mi_weight * mi_loss + lambda_mod * modularity_loss
        
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()
        total_classify_loss += classify_loss.item()
        total_mi_loss += mi_loss.item()
        total_modularity_loss += modularity_loss.item()
        
        pred = subgraph_logits.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
        total += batch.y.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100. * correct / total:.1f}%'
        })
        
        if use_wandb and wandb.run and batch_idx % 10 == 0: 
            wandb.log({
                "batch/train_loss": loss.item(),
                "batch/train_classify_loss": classify_loss.item(),
                "batch/train_mi_loss": mi_loss.item(),
                "batch/train_modularity_loss": modularity_loss.item(),
                "batch/train_acc": correct / total
            })
    
    pbar.close()
    avg_loss = total_loss / len(dataloader)
    avg_classify_loss = total_classify_loss / len(dataloader)
    avg_mi_loss = total_mi_loss / len(dataloader)
    avg_modularity_loss = total_modularity_loss / len(dataloader)
    accuracy = correct / total
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    _, f1, mcc = calc_performance_statistics(all_preds, all_labels)
    
    return avg_loss, avg_classify_loss, avg_mi_loss, avg_modularity_loss, accuracy, f1, mcc

def evaluate_model(model, sg_model, dataloader, device, epoch_num, phase="Val", training_config=None):
    model.eval()
    sg_model.eval()
    
    total_loss = 0.0
    total_classify_loss = 0.0
    total_mi_loss = 0.0
    total_modularity_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    mi_weight = training_config.get('mi_weight', 0.01) 
    lambda_mod = training_config.get('lambda_mod', 0.5)  
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num:3d} [{phase}]", 
                leave=False, file=sys.stdout, ncols=100)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            original_graph_embeddings, _, _ = model(batch)
            
            batch_subgraph = batch.clone()
            edge_mask = sg_model(batch_subgraph)
            
            batch_subgraph.edge_weight = edge_mask  
            
            subgraph_embeddings, subgraph_logits, subgraph_P = model(batch_subgraph)
            
            classify_loss = F.cross_entropy(subgraph_logits, batch.y) 
            modularity_loss = calculate_differentiable_modularity_loss(batch_subgraph, subgraph_P)
            
            if mi_weight > 0:
                sigma = training_config.get('sigma', None)
                mi_loss = CS_QMI_normalized(original_graph_embeddings, subgraph_embeddings, sigma)
            else:
                mi_loss = torch.tensor(0.0, device=batch.y.device)
                
            loss = classify_loss + mi_weight * mi_loss + lambda_mod * modularity_loss
            
            total_loss += loss.item()
            total_classify_loss += classify_loss.item()
            total_mi_loss += mi_loss.item()
            total_modularity_loss += modularity_loss.item()
            pred = subgraph_logits.argmax(dim=1)
            correct += pred.eq(batch.y).sum().item()
            total += batch.y.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100. * correct / total:.1f}%'
            })
    
    pbar.close()
    avg_loss = total_loss / len(dataloader)
    avg_classify_loss = total_classify_loss / len(dataloader)
    avg_mi_loss = total_mi_loss / len(dataloader)
    avg_modularity_loss = total_modularity_loss / len(dataloader)
    accuracy = correct / total
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    _, f1, mcc = calc_performance_statistics(all_preds, all_labels)
    
    return avg_loss, avg_classify_loss, avg_mi_loss, avg_modularity_loss, accuracy, f1, mcc

def fixed_training_and_evaluation(dataset, config, device):
    """Fixed training using 8:1:1 split (train:val:test)"""
    training_config = config["TRAINING_CONFIG"]
    wandb_config = config["WANDB_CONFIG"]
    data_config = config["DATA_CONFIG"]
    
    logger.info("\n" + "="*20 + " Fixed Training (8:1:1 Split) " + "="*20)
    
    # Initialize WandB for final training if enabled
    use_wandb_final = wandb_config.get("use_wandb", False)
    
    if use_wandb_final:
        wandb.init(
            project=wandb_config.get("project_name", "IGModel") + "_Fixed",
            entity=wandb_config.get("entity_name", None),
            name=wandb_config.get("run_name", "IGModel_Fixed") + f"_{data_config['dataset_name']}",
            config=config,
            tags=["fixed_training", data_config['dataset_name']]
        )
    
    train_loader, val_loader, test_loader = get_fixed_dataloaders(dataset, training_config, data_config, mode='fixed_811')
    
    model = IGModel(config).to(device)
    sg_model = MLP_subgraph(device, node_feature_dim=data_config['num_features']).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in sg_model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    main_model_lr = training_config.get('main_model_lr', training_config.get('learning_rate', 0.001))
    sg_model_lr = training_config.get('sg_model_lr', training_config.get('learning_rate', 0.001))
    
    optimizer_params = [
        {'params': model.parameters(), 'lr': main_model_lr, 'name': 'main_model'},
        {'params': sg_model.parameters(), 'lr': sg_model_lr, 'name': 'sg_model'}
    ]
    
    optimizer = optim.AdamW(
        optimizer_params,
        weight_decay=training_config.get('weight_decay', 0)
    )
    
    logger.info(f"Using separate learning rates - Main model: {main_model_lr}, SG model: {sg_model_lr}")
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=training_config.get('scheduler_factor', 0.8),
        patience=training_config.get('scheduler_patience', 8),
        min_lr=float(training_config.get('scheduler_min_lr', 5e-7))
    )

    best_val_acc = 0.0
    best_val_metrics = None
    best_val_epoch = 0
    
    best_model_state = None
    best_sg_model_state = None
    
    best_val_loss = float('inf')
    use_early_stopping = training_config.get('early_stopping', True)
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    early_stopping_counter = 0
    
    for epoch in range(1, training_config['epochs'] + 1):
        train_loss, train_cls_loss, train_mi_loss, train_mod_loss, train_acc, train_f1, train_mcc = train_one_epoch(
            model, sg_model, train_loader, optimizer, device, epoch,
            training_config, use_wandb_final
        )
        
        val_loss, val_cls_loss, val_mi_loss, val_mod_loss, val_acc, val_f1, val_mcc = evaluate_model(
            model, sg_model, val_loader, device, epoch, "Val", training_config
        )
        
        scheduler.step(val_loss)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_metrics = (val_loss, val_cls_loss, val_mi_loss, val_mod_loss, val_acc, val_f1, val_mcc)
            best_val_epoch = epoch
            best_model_state = model.state_dict().copy()
            best_sg_model_state = sg_model.state_dict().copy()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if use_early_stopping:
                early_stopping_counter = 0
        else:
            if use_early_stopping:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch} epochs (based on validation loss)")
                    break
        
        if use_wandb_final and wandb.run:
            wandb_log_dict = {
                "epoch": epoch,
                "learning_rate/main_model": optimizer.param_groups[0]['lr'],
                "learning_rate/sg_model": optimizer.param_groups[1]['lr'],
                "train/total_loss": train_loss,
                "train/classify_loss": train_cls_loss,
                "train/mi_loss": train_mi_loss,
                "train/modularity_loss": train_mod_loss,
                "train/accuracy": train_acc,
                "train/f1_score": train_f1,
                "train/mcc": train_mcc,
                "validation/total_loss": val_loss,
                "validation/classify_loss": val_cls_loss,
                "validation/mi_loss": val_mi_loss,
                "validation/modularity_loss": val_mod_loss,
                "validation/accuracy": val_acc,
                "validation/f1_score": val_f1,
                "validation/mcc": val_mcc,
            }
            wandb.log(wandb_log_dict)
        
        logger.info(f"Epoch {epoch:3d} | Train: {train_acc*100:5.1f}% | Val: {val_acc*100:5.1f}%")
        logger.info(f"         ├─ Train Loss: cls={train_cls_loss:.3f}, mi={train_mi_loss:.4f}, mod={train_mod_loss:.4f}")
        logger.info(f"         ├─ Val Loss:   cls={val_cls_loss:.3f}")
        logger.info(f"         └─ Val Acc:    {val_acc*100:5.1f}%")
        
    logger.info(f"\n{'='*60}")
    logger.info(f"Training Completed! Best model from epoch {best_val_epoch}")

    model.load_state_dict(best_model_state)
    sg_model.load_state_dict(best_sg_model_state)
    
    test_loss, test_cls_loss, test_mi_loss, test_mod_loss, test_acc, test_f1, test_mcc = evaluate_model(
        model, sg_model, test_loader, device, best_val_epoch, "Test", training_config
    )
    
    logger.info(f"\nBest Results (Epoch {best_val_epoch}):")
    logger.info(f"  Accuracy: {test_acc*100:.2f}%")
    logger.info(f"  F1-Score: {test_f1:.4f}")
    logger.info(f"  MCC:      {test_mcc:.4f}")
    logger.info(f"  Loss:     {test_cls_loss:.4f}")
    logger.info(f"{'='*60}")
    if use_wandb_final and wandb.run:
        wandb.log({
            "best/epoch": best_val_epoch,
            "best/test_accuracy": test_acc,
            "best/test_f1": test_f1,
            "best/test_mcc": test_mcc,
            "best/test_loss": test_cls_loss
        })
    
    result = {
        'best_epoch': best_val_epoch,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_mcc': test_mcc,
        'test_loss': test_cls_loss
    }
    
    if use_wandb_final:
        wandb.finish()
    
    return result

def main(cli_args=None, config=None, mode=None):
    if config is None:
        if cli_args and hasattr(cli_args, 'dataset') and cli_args.dataset in DATASET_CONFIGS:
            config_path = DATASET_CONFIGS[cli_args.dataset]
        else:
            config_path = None
        config = load_configs(config_path, cli_args)
    
    training_config = config["TRAINING_CONFIG"]
    wandb_config = config["WANDB_CONFIG"]
    data_config = config["DATA_CONFIG"]
    
    if mode is None and cli_args is not None:
        mode = getattr(cli_args, 'mode', 'fixed')
    elif mode is None:
        mode = 'sweep'
    
    dataset = load_dataset(data_config)
    
    seed = training_config.get('seed', 9)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device(training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    logger.info(f"Dataset: {data_config['dataset_name']}")
    logger.info(f"Number of graphs: {len(dataset)}")
    logger.info(f"Number of classes: {data_config['num_classes']}")
    logger.info(f"Number of node features: {data_config['num_features']}")
    logger.info(f"Execution mode: {mode}")
    logger.info(f"Global training seed: {seed}")
    logger.info(f"Dataset split seed: {data_config.get('dataset_seed', 42)}")
    
    if mode == 'sweep':
        logger.info("Starting hyperparameter tuning phase...")
        best_val_metrics = hyperparameter_tuning(dataset, config, device)
        val_loss, val_cls_loss, val_mi_loss, val_mod_loss, val_acc, val_f1, val_mcc = best_val_metrics
        
        if wandb_config.get("use_wandb", False) and wandb.run:
            wandb.log({
                "best_validation/accuracy": val_acc,
                "best_validation/f1_score": val_f1,
                "best_validation/mcc": val_mcc,
                "best_validation/classify_loss": val_cls_loss
            })
        
        return {
            'val_acc': val_acc, 
            'val_f1': val_f1,
            'val_mcc': val_mcc,
            'val_loss': val_loss
        }
        
    elif mode == 'fixed':
        logger.info("Starting fixed training and evaluation phase...")
        fixed_results = fixed_training_and_evaluation(dataset, config, device)
        return fixed_results   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGMamba Graph Classification")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset name")
    parser.add_argument("--mode", type=str, default="fixed", 
                        choices=["sweep", "fixed"], 
                        help="Run mode: sweep, fixed")
    parser.add_argument("--num_sweep_runs", type=int, default=50, 
                        help="Number of sweep runs")
    args = parser.parse_args()
    
    base_config = load_configs(DATASET_CONFIGS[args.dataset], args)
    base_config["DATA_CONFIG"]["dataset_name"] = args.dataset
    
    if args.mode == 'sweep':
        def sweep_train():
            run = wandb.init()
            
            wandb.run.log_code(".", include_fn=lambda path: path.endswith(('.py', '.yaml', '.yml', '.txt', '.md')))
            print(" Code saved to WandB")
            
            sweep_params = dict(wandb.config)
            
            sweep_config_path = f"sweeps/{args.dataset}_sweep.yaml"
            tuner = HyperparameterTuner(base_config, sweep_config_path)
            updated_config = tuner.update_config(base_config, sweep_params)
            
            results = main(cli_args=None, config=updated_config, mode='sweep')
            wandb.log({"val_acc_target": results.get('val_acc', 0)})
            wandb.finish()
        
        sweep_config_path = f"sweeps/{args.dataset}_sweep.yaml"
        tuner = HyperparameterTuner(base_config, sweep_config_path)
        project_name = f"{base_config['WANDB_CONFIG']['project_name']}_{args.dataset}"
        
        sweep_id = tuner.run_sweep(
            train_function=sweep_train,
            project_name=project_name,
            num_runs=args.num_sweep_runs
        )
        print(f"Sweep completed with ID: {sweep_id}")
        
    else:
        print(f"Starting {args.mode} training for dataset: {args.dataset}")
        results = main(cli_args=args, config=base_config, mode=args.mode)
        print("Training completed successfully!") 