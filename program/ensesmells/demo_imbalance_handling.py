"""
Demo script showing how to use the new imbalanced data handling techniques

This script demonstrates:
1. Class distribution analysis
2. Automatic weight calculation
3. Different loss function configurations
4. Result comparison
"""

import torch
import numpy as np
from focal_loss import FocalLoss, WeightedFocalLoss, calculate_class_weights, get_loss_function
from imbalance_utils import analyze_class_distribution, recommend_loss_config

def demo_class_analysis():
    """Demo class distribution analysis"""
    print("=== Demo: Class Distribution Analysis ===")
    
    # Simulate imbalanced dataset
    # 10% positive, 90% negative samples
    n_samples = 1000
    n_positive = 100
    n_negative = 900
    
    # Create synthetic labels
    labels = [1] * n_positive + [0] * n_negative
    labels = np.random.permutation(labels)  # Shuffle
    
    # Analyze distribution
    stats = analyze_class_distribution(labels, "Synthetic Imbalanced Dataset")
    
    # Get recommendations
    recommendations = recommend_loss_config(stats)
    
    return stats, recommendations

def demo_loss_functions():
    """Demo different loss functions"""
    print("\n=== Demo: Loss Function Comparison ===")
    
    # Simulate some predictions and targets
    batch_size = 32
    predictions = torch.randn(batch_size, 1)  # Raw logits
    targets = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Calculate pos_weight
    pos_weight = calculate_class_weights(targets.flatten())
    print(f"Calculated pos_weight: {pos_weight.item():.4f}")
    
    # Test different loss functions
    loss_configs = [
        {'type': 'bce', 'pos_weight': 1.0, 'alpha': None, 'gamma': None},
        {'type': 'bce', 'pos_weight': pos_weight.item(), 'alpha': None, 'gamma': None},
        {'type': 'focal', 'pos_weight': 1.0, 'alpha': 0.25, 'gamma': 2.0},
        {'type': 'focal', 'pos_weight': pos_weight.item(), 'alpha': 0.25, 'gamma': 2.0},
        {'type': 'weighted_focal', 'pos_weight': pos_weight.item(), 'alpha': 0.25, 'gamma': 2.0},
    ]
    
    print(f"\nLoss function comparison:")
    print(f"{'Loss Type':<20} {'Pos Weight':<12} {'Alpha':<8} {'Gamma':<8} {'Loss Value':<12}")
    print("-" * 70)
    
    for config in loss_configs:
        loss_fn = get_loss_function(
            loss_type=config['type'],
            pos_weight=torch.tensor(config['pos_weight']) if config['pos_weight'] else None,
            alpha=config['alpha'],
            gamma=config['gamma']
        )
        
        with torch.no_grad():
            loss_value = loss_fn(predictions, targets)
        
        print(f"{config['type']:<20} {config['pos_weight'] or 'None':<12} "
              f"{config['alpha'] or 'None':<8} {config['gamma'] or 'None':<8} "
              f"{loss_value.item():<12.4f}")

def demo_training_setup():
    """Demo how to set up training with new loss functions"""
    print("\n=== Demo: Training Setup ===")
    
    # Example of how to integrate into training loop
    code_example = """
# Example integration into your training code:

# 1. Analyze your dataset
train_labels = [...]  # Your training labels
stats = analyze_class_distribution(train_labels, "Your Dataset")
recommendations = recommend_loss_config(stats)

# 2. Choose loss function based on recommendations
loss_config = recommendations[0]  # Use first recommendation

# 3. Create loss function
pos_weight = torch.tensor(loss_config['pos_weight']) if loss_config['pos_weight'] else None
train_loss_fn = get_loss_function(
    loss_type=loss_config['type'],
    pos_weight=pos_weight,
    alpha=loss_config['alpha'],
    gamma=loss_config['gamma']
)

# 4. Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        predictions = model(batch)
        loss = train_loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
"""
    print(code_example)

if __name__ == "__main__":
    # Run demos
    demo_class_analysis()
    demo_loss_functions()
    demo_training_setup()
    
    print("\n=== Summary ===")
    print("✅ Class distribution analysis implemented")
    print("✅ Automatic pos_weight calculation implemented") 
    print("✅ Focal Loss implemented")
    print("✅ Weighted Focal Loss implemented")
    print("✅ Loss function factory implemented")
    print("✅ Training integration complete")
    print("\nYou can now run your main training script to test different loss configurations!")
