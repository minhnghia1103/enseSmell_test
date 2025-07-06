import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for binary classification
    
    Args:
        alpha: Weighting factor for rare class (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        reduction: Specifies the reduction to apply to the output
        pos_weight: Weight for positive class to handle class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        """
        # Convert logits to probabilities
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss that combines class weights with focal loss
    """
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        sigmoid_p = torch.sigmoid(inputs)
        
        # Calculate cross entropy manually for better control
        ce_loss = -(targets * torch.log(sigmoid_p + 1e-8) + 
                   (1 - targets) * torch.log(1 - sigmoid_p + 1e-8))
        
        # Apply class weights (pos_weight)
        if self.pos_weight is not None:
            ce_loss = ce_loss * (targets * self.pos_weight + (1 - targets))
        
        # Calculate pt for focal term
        pt = targets * sigmoid_p + (1 - targets) * (1 - sigmoid_p)
        
        # Calculate focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha term
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def calculate_class_weights(targets, method='inverse_freq'):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        targets: List or tensor of target labels
        method: Method to calculate weights ('inverse_freq', 'balanced')
    
    Returns:
        pos_weight: Weight for positive class
    """
    if isinstance(targets, list):
        targets = torch.tensor(targets, dtype=torch.float)
    elif isinstance(targets, torch.Tensor):
        targets = targets.float()
    else:
        targets = torch.tensor(targets, dtype=torch.float)
    
    # Count positive and negative samples
    pos_count = torch.sum(torch.abs(targets - 1.0) < 1e-6).float()  # Close to 1.0
    neg_count = torch.sum(torch.abs(targets - 0.0) < 1e-6).float()  # Close to 0.0
    total_count = len(targets)
    
    # Handle edge cases
    if pos_count == 0:
        return torch.tensor(1.0)
    if neg_count == 0:
        return torch.tensor(1.0)
    
    if method == 'inverse_freq':
        # Inverse frequency weighting
        pos_weight = neg_count / pos_count
    elif method == 'balanced':
        # Balanced class weighting
        pos_weight = total_count / (2.0 * pos_count)
    else:
        pos_weight = torch.tensor(1.0)
    
    # Cap extreme weights to prevent training instability
    pos_weight = torch.clamp(pos_weight, min=0.1, max=100.0)
    
    return pos_weight

def get_loss_function(loss_type='focal', pos_weight=None, alpha=0.25, gamma=2.0):
    """
    Factory function to get different loss functions
    
    Args:
        loss_type: Type of loss ('bce', 'focal', 'weighted_focal')
        pos_weight: Weight for positive class
        alpha: Alpha parameter for focal loss
        gamma: Gamma parameter for focal loss
    
    Returns:
        Loss function
    """
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    elif loss_type == 'weighted_focal':
        return WeightedFocalLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
