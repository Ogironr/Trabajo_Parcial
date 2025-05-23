import torch
import torch.nn as nn

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=3, dropout=0.5):
        super(ImprovedNeuralNetwork, self).__init__()
        
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Crear capas ocultas
        self.layers = nn.ModuleList()
        
        current_size = input_size
        for _ in range(n_layers):
            layer_block = nn.Sequential(
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer_block)
            current_size = hidden_size
        
        # Capa de salida
        self.output_fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.input_bn(x)
        
        # Pasar por cada capa
        for layer in self.layers:
            x = layer(x)
        
        # Capa de salida
        x = self.output_fc(x)
        x = self.sigmoid(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return False

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6))
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_loss = ce_loss * ((1 - pt) ** self.gamma)
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=3.0, focal_weight=0.8, dice_weight=0.2):
        super().__init__()
        self.focal_loss = FocalLoss(gamma)
        self.dice_loss = DiceLoss()
        self.alpha = alpha
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred, target):
        # Focal Loss con peso de clase
        focal = self.focal_loss(pred, target)
        focal_weighted = torch.where(target == 1, 
                                   self.alpha * focal, 
                                   (1 - self.alpha) * focal)
        
        # Dice Loss
        dice = self.dice_loss(pred, target)
        
        # Pérdida combinada
        combined_loss = self.focal_weight * focal_weighted.mean() + \
                       self.dice_weight * dice
        
        return combined_loss
