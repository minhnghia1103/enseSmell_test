import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_distribution(targets, dataset_name="Dataset"):
    """
    Analyze and visualize class distribution in the dataset
    
    Args:
        targets: List or tensor of target labels
        dataset_name: Name of the dataset for display
    
    Returns:
        Dict with class distribution statistics
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    
    # Flatten if needed
    if len(targets.shape) > 1:
        targets = targets.flatten()
    
    # Convert to float for consistent comparison
    targets = targets.astype(float)
    
    # Count classes - handle both 0/1 and 0.0/1.0
    counter = Counter(targets)
    total_samples = len(targets)
    
    # Calculate statistics - be more flexible with class labels
    pos_count = 0
    neg_count = 0
    
    for label, count in counter.items():
        if abs(label - 1.0) < 1e-6:  # Close to 1.0
            pos_count += count
        elif abs(label - 0.0) < 1e-6:  # Close to 0.0
            neg_count += count
    
    # Handle edge cases
    if total_samples == 0:
        return {
            'total_samples': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'imbalance_ratio': 1.0,
            'recommended_pos_weight': 1.0
        }
    
    pos_ratio = pos_count / total_samples
    neg_ratio = neg_count / total_samples
    imbalance_ratio = max(pos_count, neg_count) / max(min(pos_count, neg_count), 1) if min(pos_count, neg_count) > 0 else float('inf')
    
    stats = {
        'total_samples': total_samples,
        'positive_samples': pos_count,
        'negative_samples': neg_count,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio,
        'imbalance_ratio': imbalance_ratio,
        'recommended_pos_weight': neg_count / max(pos_count, 1) if pos_count > 0 else 1.0
    }
    
    # Print statistics
    print(f"\n=== Class Distribution Analysis for {dataset_name} ===")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {pos_count} ({pos_ratio:.2%})")
    print(f"Negative samples: {neg_count} ({neg_ratio:.2%})")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"Recommended pos_weight: {stats['recommended_pos_weight']:.4f}")
    
    return stats

def recommend_loss_config(class_stats):
    """
    Recommend loss function configuration based on class distribution
    
    Args:
        class_stats: Output from analyze_class_distribution
    
    Returns:
        Dict with recommended configurations
    """
    imbalance_ratio = class_stats['imbalance_ratio']
    pos_weight = class_stats['recommended_pos_weight']
    
    recommendations = []
    
    if imbalance_ratio < 2:
        # Mild imbalance
        recommendations.append({
            'type': 'bce',
            'pos_weight': 1.0,
            'alpha': None,
            'gamma': None,
            'reason': 'Mild imbalance - standard BCE should work'
        })
    elif imbalance_ratio < 5:
        # Moderate imbalance
        recommendations.extend([
            {
                'type': 'bce',
                'pos_weight': pos_weight,
                'alpha': None,
                'gamma': None,
                'reason': 'Moderate imbalance - weighted BCE'
            },
            {
                'type': 'focal',
                'pos_weight': 1.0,
                'alpha': 0.25,
                'gamma': 2.0,
                'reason': 'Moderate imbalance - focal loss'
            }
        ])
    else:
        # Severe imbalance
        recommendations.extend([
            {
                'type': 'focal',
                'pos_weight': min(pos_weight, 10.0),  # Cap extreme weights
                'alpha': 0.25,
                'gamma': 2.0,
                'reason': 'Severe imbalance - focal loss with pos_weight'
            },
            {
                'type': 'weighted_focal',
                'pos_weight': min(pos_weight, 10.0),
                'alpha': 0.5,
                'gamma': 2.0,
                'reason': 'Severe imbalance - weighted focal loss'
            },
            {
                'type': 'focal',
                'pos_weight': 1.0,
                'alpha': 0.75,
                'gamma': 3.0,
                'reason': 'Severe imbalance - high alpha focal loss'
            }
        ])
    
    print(f"\n=== Recommended Loss Configurations ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['type'].upper()}: pos_weight={rec['pos_weight']}, "
              f"alpha={rec['alpha']}, gamma={rec['gamma']}")
        print(f"   Reason: {rec['reason']}")
    
    return recommendations

def compare_loss_configurations(results_dict):
    """
    Compare different loss configuration results
    
    Args:
        results_dict: Dictionary with configuration names as keys and metrics as values
                     Format: {config_name: [precision, recall, f1, auc, mcc]}
    """
    print(f"\n=== Loss Configuration Comparison ===")
    print(f"{'Configuration':<50} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8} {'MCC':<8}")
    print("=" * 90)
    
    best_f1 = 0
    best_config = None
    
    for config, metrics in results_dict.items():
        precision, recall, f1, auc, mcc = metrics
        print(f"{config:<50} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f} {auc:<8.4f} {mcc:<8.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = config
    
    print("=" * 90)
    print(f"Best configuration: {best_config} (F1={best_f1:.4f})")
    
    return best_config, best_f1

def plot_class_distribution(targets, save_path=None):
    """
    Plot class distribution
    
    Args:
        targets: List or tensor of target labels
        save_path: Path to save the plot (optional)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)
    
    if len(targets.shape) > 1:
        targets = targets.flatten()
    
    plt.figure(figsize=(8, 6))
    unique, counts = np.unique(targets, return_counts=True)
    
    plt.bar(['Negative (0)', 'Positive (1)'], counts, color=['skyblue', 'lightcoral'])
    plt.title('Class Distribution')
    plt.ylabel('Number of Samples')
    plt.xlabel('Class')
    
    # Add percentage labels
    total = sum(counts)
    for i, count in enumerate(counts):
        plt.text(i, count + total*0.01, f'{count}\n({count/total:.1%})', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
