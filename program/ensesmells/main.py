# lib pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# lib torch_geometric for handling graph data
try:
    from torch_geometric.data import Batch
except ImportError:
    # Fallback for older versions
    try:
        from torch_geometric.loader import Batch
    except ImportError:
        # If torch_geometric is not available, create a dummy Batch
        class Batch:
            @staticmethod
            def from_data_list(data_list):
                # Simple fallback - just return the first item
                return data_list[0] if data_list else None

# lib default python
import os
import random
import time
import gc

# lib science math
import numpy as np

# lib sklearn
from sklearn.model_selection import StratifiedKFold, KFold

# my class
import config
import utils
import data
import train
from model import CNN_LSTM, CNN_BiLSTM, CNN_LSTM_METRICS, calculate_size_lstm
from focal_loss import FocalLoss, WeightedFocalLoss, calculate_class_weights, get_loss_function
from imbalance_utils import analyze_class_distribution, recommend_loss_config, compare_loss_configurations
import time
import datetime
from config import argument
# my tool
from utils import write_file
import json

args = argument()

def custom_collate_fn(batch):
    """
    Custom collate function to handle torch_geometric Data objects and mixed data types
    Ensures all tensors are on the same device
    """
    if not batch:
        return None
    
    # Import config to get device
    import config
    device = config.Config.DEVICE
    
    # Check the structure of the first item to understand the batch format
    first_item = batch[0]
    
    if len(first_item) == 2:  # (data, label)
        data_list = []
        label_list = []
        
        for item in batch:
            data, label = item
            # Check if data contains graph data
            if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                # This is graph data, move to device
                data = data.to(device)
                data_list.append(data)
            else:
                # Regular tensor data, move to device
                data = data.to(device)
                data_list.append(data)
            
            # Move label to device
            label = label.to(device)
            label_list.append(label)
        
        # Handle batching based on data type
        if hasattr(data_list[0], 'x') and hasattr(data_list[0], 'edge_index'):
            # Graph data
            batched_data = Batch.from_data_list(data_list)
        else:
            # Regular tensor data
            batched_data = torch.stack(data_list)
        
        batched_labels = torch.stack(label_list)
        return batched_data, batched_labels
        
    elif len(first_item) == 4:  # (data, metrics, label, graph)
        data_list = []
        metrics_list = []
        label_list = []
        graph_list = []
        
        for item in batch:
            data, metrics, label, graph = item
            
            # Move all tensors to device
            data = data.to(device)
            metrics = metrics.to(device)
            label = label.to(device)
            graph = graph.to(device)
            
            data_list.append(data)
            metrics_list.append(metrics)
            label_list.append(label)
            graph_list.append(graph)
        
        batched_data = torch.stack(data_list)
        batched_metrics = torch.stack(metrics_list)
        batched_labels = torch.stack(label_list)
        batched_graphs = Batch.from_data_list(graph_list)
        
        return batched_data, batched_metrics, batched_labels, batched_graphs
    
    else:
        # Fallback to default collate for other cases
        try:
            result = torch.utils.data.dataloader.default_collate(batch)
            # Move result to device if it's a tensor
            if isinstance(result, torch.Tensor):
                result = result.to(device)
            elif isinstance(result, (list, tuple)):
                result = tuple(item.to(device) if isinstance(item, torch.Tensor) else item for item in result)
            return result
        except:
            # If default collate fails, return batch as is
            return batch

if __name__ == "__main__":
    # Configuration for different loss functions and their parameters
    loss_configs = [
        {'type': 'bce', 'pos_weight': 1.0, 'alpha': None, 'gamma': None},
        {'type': 'bce', 'pos_weight': 2.0, 'alpha': None, 'gamma': None},
        {'type': 'bce', 'pos_weight': 4.0, 'alpha': None, 'gamma': None},
        {'type': 'focal', 'pos_weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        {'type': 'focal', 'pos_weight': 2.0, 'alpha': 0.25, 'gamma': 2.0},
        {'type': 'focal', 'pos_weight': 1.0, 'alpha': 0.5, 'gamma': 2.0},
        {'type': 'focal', 'pos_weight': 1.0, 'alpha': 0.25, 'gamma': 1.5},
        {'type': 'weighted_focal', 'pos_weight': 2.0, 'alpha': 0.25, 'gamma': 2.0},
    ]

    kernel_size_set = [3, 4, 5, 6, 7]
    now = datetime.datetime.now()

    data_path = args.data_path
    graph_path = args.graph_path
    vocab_path = args.vocab_path
    with open(vocab_path, 'r') as f:
        vocabdict = json.load(f)
    vocab_size = len(vocabdict)

    result_summary = {}

    for loss_config in loss_configs:
        for kernel_size in kernel_size_set:
            smell, model = utils.get_smell_and_model(args.data_path)   
            
            # Create descriptive filename based on loss configuration
            loss_type = loss_config['type']
            pos_weight = loss_config['pos_weight']
            alpha = loss_config.get('alpha', 'None')
            gamma = loss_config.get('gamma', 'None')
            
            file_name = f'{model}_{smell}_{now.strftime("%d%m%Y_%H%M")}_loss_{loss_type}_pw_{pos_weight}_alpha_{alpha}_gamma_{gamma}_kernel_{kernel_size}'
            track_file = f'{args.tracking_dir}/{file_name}.txt'
            result_file = f'{args.result_dir}/{model}_{smell}_{now.strftime("%d%m%Y_%H%M")}.txt'
            
            precision = []
            recall = []
            f1 = []
            auc = []
            mcc = []
            
            if args.model == 'EnseSmells':
                load_dataset = enumerate(utils.get_CombineData_pickle(data_path))
            elif args.model == 'DeepSmells_TokenIndexing':
                load_dataset = enumerate(utils.get_data_token_indexing(data_path))
            elif args.model == 'EnseSmell_TokenIndexing' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                load_dataset = enumerate(utils.get_data_token_indexing_COMBINING(data_path, graph_path))
            else: # DeepSmells
                load_dataset = enumerate(utils.get_data_pickle(data_path))

            for index, datasets in load_dataset:
                print(f"{'=+'*25} FOLD: {index+1} / 5 {'+='*25}")
                write_file(track_file, f"{'=+'*25} FOLD: {index+1} / 5 {'+='*25}\n")
                
                # Analyze class distribution for current fold
                if index == 0:  # Only analyze for first fold to avoid repetition
                    train_targets_flat = [datasets.train_labels[i][0] for i in range(len(datasets.train_labels))]
                    class_stats = analyze_class_distribution(train_targets_flat, f"Training Set - Fold {index+1}")
                    write_file(track_file, f"Class distribution analysis:\n")
                    write_file(track_file, f"Positive samples: {class_stats['positive_samples']} ({class_stats['positive_ratio']:.2%})\n")
                    write_file(track_file, f"Negative samples: {class_stats['negative_samples']} ({class_stats['negative_ratio']:.2%})\n")
                    write_file(track_file, f"Imbalance ratio: {class_stats['imbalance_ratio']:.2f}:1\n")
                    write_file(track_file, f"Recommended pos_weight: {class_stats['recommended_pos_weight']:.4f}\n\n")
                
                # Sample elements randomly from given list of ids, no replacement
                if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                    train_set = data.DatasetCombineWithGraph(datasets.train_data, datasets.train_data_metrics, datasets.train_labels, datasets.train_graph_data)
                    valid_set = data.DatasetCombineWithGraph(datasets.eval_data, datasets.eval_data_metrics, datasets.eval_labels, datasets.eval_graph_data)
                if args.model == 'DeepSmells' or args.model == 'DeepSmells_TokenIndexing': 
                    train_set = data.Dataset(datasets.train_data, datasets.train_labels)
                    valid_set = data.Dataset(datasets.eval_data, datasets.eval_labels)

                # Define data loaders for training and testing data in this fold
                # Use custom collate function only if we detect graph data
                if args.model in ['DeepSmells_METRICS', 'DeepSmells_TokenIndexing_METRICS']:
                    # These models use graph data, so use custom collate function
                    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, collate_fn=custom_collate_fn)
                    valid_loader = DataLoader(valid_set, batch_size=args.valid_batchsize, shuffle=True, collate_fn=custom_collate_fn)
                else:
                    # Regular models without graph data
                    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True)
                    valid_loader = DataLoader(valid_set, batch_size=args.valid_batchsize, shuffle=True)

                length_code = train_set[0][0].size()[1]
                # Calculate size LSTM - CC
                input_size_lstm = calculate_size_lstm(input_size=length_code, kernel_size=kernel_size)

                # Initialize the model, optimizer, scheduler, loss
                if args.model == 'DeepSmells' or args.model == 'DeepSmells_TokenIndexing':
                    model = CNN_LSTM(kernel_size = kernel_size, input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm).to(config.Config.DEVICE)
                if args.model == 'DeepSmells-BiLSTM':
                    model = CNN_BiLSTM(kernel_size = kernel_size, input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm).to(config.Config.DEVICE)
                if args.model == 'DeepSmells_METRICS' or args.model == 'DeepSmells_TokenIndexing_METRICS':
                    metrics_size = train_set[0][1].size()[0]
                    model = CNN_LSTM_METRICS(kernel_size = kernel_size, vocab_size=vocab_size, embedding_dim=128,input_size_lstm=input_size_lstm, hidden_size_lstm=args.hidden_size_lstm, metrics_size = metrics_size).to(config.Config.DEVICE)
                optimizer = optim.SGD(model.parameters(), lr=args.lr)
                # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

                # Calculate class weights automatically from training data
                train_targets = [datasets.train_labels[i][0] for i in range(len(datasets.train_labels))]
                auto_pos_weight = calculate_class_weights(train_targets, method='inverse_freq')
                
                # Use specified pos_weight or auto-calculated weight
                final_pos_weight = torch.tensor(pos_weight, dtype=torch.float) if pos_weight != 'auto' else auto_pos_weight
                
                # Create loss functions based on configuration
                train_loss_fn = get_loss_function(
                    loss_type=loss_type,
                    pos_weight=final_pos_weight,
                    alpha=alpha if alpha != 'None' else None,
                    gamma=gamma if gamma != 'None' else None
                ).to(config.Config.DEVICE)
                
                valid_loss_fn = get_loss_function(
                    loss_type=loss_type,
                    pos_weight=final_pos_weight,
                    alpha=alpha if alpha != 'None' else None,
                    gamma=gamma if gamma != 'None' else None
                ).to(config.Config.DEVICE)
                
                print(f"Using {loss_type} loss with pos_weight={final_pos_weight.item():.4f}")
                write_file(track_file, f"Using {loss_type} loss with pos_weight={final_pos_weight.item():.4f}\n")
                
                trainer = train.Trainer(
                    device = config.Config.DEVICE,
                    dataloader = (train_loader, valid_loader),
                    model = model,
                    loss_fns = (train_loss_fn, valid_loss_fn),
                    optimizer = optimizer,
                    # scheduler = step_lr_scheduler,
                )

                best_precision, best_recall, best_f1, best_auc, best_mcc = trainer.fit(
                    epochs = args.nb_epochs,
                    checkpoint_dir = None,
                    custom_name = file_name,
                    track_dir = track_file,
                    threshold = args.threshold,
                )

                precision.append(best_precision)
                recall.append(best_recall)
                f1.append(best_f1)
                auc.append(best_auc)
                mcc.append(best_mcc)
                # del model, optimizer, train_loss_fn, valid_loss_fn, trainer, best_precision, best_recall, best_f1, best_mcc
                gc.collect()
            
            config_key = f"loss_{loss_type}_pw_{pos_weight}_alpha_{alpha}_gamma_{gamma}_kernel_{kernel_size}"
            result_summary[config_key] = [sum(precision)/len(precision), 
                                        sum(recall)/len(recall), 
                                        sum(f1)/len(f1), 
                                        sum(auc)/len(auc), 
                                        sum(mcc)/len(mcc)                                                                                                                                        
                                      ]
            write_file(result_file, f"{config_key},{sum(precision)/len(precision)},{sum(recall)/len(recall)},{sum(f1)/len(f1)},{sum(auc)/len(auc)},{sum(mcc)/len(mcc)}\n")
    # Find and display best configuration
    best_config, best_f1 = compare_loss_configurations(result_summary)
    write_file(result_file, f"BEST-{best_config},{result_summary[best_config][0]}, {result_summary[best_config][1]}, {result_summary[best_config][2]}, {result_summary[best_config][3]}, {result_summary[best_config][4]}\n")
