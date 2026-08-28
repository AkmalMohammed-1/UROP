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
    
    # 1. PULL: Feature-space MSE (match the feature means of the batches)
    loss_mse = F.mse_loss(feat_syn.mean(dim=0), feat_real.detach().mean(dim=0))
    
    # 2. PULL: Semantic-space Calibration (Cross-Entropy)
    # Cross entropy works per-sample, so no averaging needed before the loss
    loss_calib = F.cross_entropy(logits_syn, label_syn)
    
    # 3. PUSH: Divergence Loss (Reverse KL)
    # We want D_KL(q(y|x_s) || p(y|x_r))
    # q is the posterior for each synthetic sample
    q = F.softmax(logits_syn, dim=1)         # shape: (batch_syn, num_classes)
    log_q = F.log_softmax(logits_syn, dim=1) # shape: (batch_syn, num_classes)
    
    # p is the average posterior over the real batch
    p_real = F.softmax(logits_real.detach(), dim=1).mean(dim=0, keepdim=True) # shape: (1, num_classes)
    log_p_real = torch.log(p_real + 1e-8)    # shape: (1, num_classes)
    
    # Compute KL divergence for each synthetic sample and average over the batch
    loss_div = torch.sum(q * (log_q - log_p_real), dim=1).mean(dim=0)
    
    lambda_mse = getattr(args, 'lambda_mse', 1.0)
    lambda_calib = getattr(args, 'lambda_calib', 1.0)
    lambda_div = getattr(args, 'lambda_div', 0.5)
    
    # Total loss: pull forces minus push force
    loss_align = (lambda_mse * loss_mse) + (lambda_calib * loss_calib)
    total_loss = loss_align - (lambda_div * loss_div)
    
    return total_loss, loss_mse, loss_calib, loss_div