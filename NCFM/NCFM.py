import torch
import torch.nn as nn
import torch.nn.functional as F

def ppdd_loss(img_real, img_syn, label_syn, model, args):
    """
    PPDD Unified Push-Pull Loss
    """
    # Get logits and features for real and synthetic images
    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)
        
    logits_syn, feat_syn = model(img_syn, return_features=True)
    
    # 1. PULL: Feature-space MSE
    loss_mse = F.mse_loss(feat_syn, feat_real.detach())
    
    # 2. PULL: Semantic-space Calibration (Cross-Entropy)
    loss_calib = F.cross_entropy(logits_syn, label_syn)
    
    # 3. PUSH: Divergence Loss (Reverse KL)
    log_q = F.log_softmax(logits_syn, dim=1)
    p = F.softmax(logits_real.detach(), dim=1)
    loss_div = F.kl_div(log_q, p, reduction='batchmean')
    
    lambda_mse = getattr(args, 'lambda_mse', 1.0)
    lambda_calib = getattr(args, 'lambda_calib', 1.0)
    lambda_div = getattr(args, 'lambda_div', 0.5)
    
    # Total loss: pull forces minus push force
    loss_align = (lambda_mse * loss_mse) + (lambda_calib * loss_calib)
    total_loss = loss_align - (lambda_div * loss_div)
    
    return total_loss, loss_mse, loss_calib, loss_div