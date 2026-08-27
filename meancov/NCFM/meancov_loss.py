import torch
import torch.nn.functional as F
from geoopt.manifolds.lorentz import Lorentz

K_CURV = 1
man = Lorentz(k=K_CURV)

def to_lorentz(feat, eps=1e-8):
    h = man.expmap0(F.pad(feat, pad=(1, 0)))
    return man.projx(h)

def l_meancov(h_real, h_syn):
    """
    Mean and Covariance matching on the tangent space of the Lorentz manifold.
    h_real : (N_r, D+1) — detached
    h_syn  : (N_s, D+1) — has gradient
    """
    N_r = h_real.shape[0]
    N_s = h_syn.shape[0]

    # Cap batch sizes for memory efficiency
    max_n = 64
    if N_r > max_n:
        idx   = torch.randperm(N_r, device=h_real.device)[:max_n]
        h_real = h_real[idx]
        N_r   = max_n
    if N_s > max_n:
        idx  = torch.randperm(N_s, device=h_syn.device)[:max_n]
        h_syn = h_syn[idx]
        N_s  = max_n

    # 1. Map to tangent space at the origin
    v_real = man.logmap0(h_real)
    v_syn = man.logmap0(h_syn)
    
    # Drop the first coordinate (which is 0 in the tangent space at the origin)
    v_real = v_real[:, 1:]
    v_syn = v_syn[:, 1:]
    
    # 2. Mean
    mu_real = v_real.mean(dim=0)
    mu_syn = v_syn.mean(dim=0)
    loss_mean = F.mse_loss(mu_syn, mu_real.detach())
    
    # 3. Covariance
    v_real_centered = v_real - mu_real.unsqueeze(0)
    v_syn_centered = v_syn - mu_syn.unsqueeze(0)
    
    cov_real = (v_real_centered.T @ v_real_centered) / (v_real.size(0) - 1 + 1e-5)
    cov_syn = (v_syn_centered.T @ v_syn_centered) / (v_syn.size(0) - 1 + 1e-5)
    
    loss_cov = F.mse_loss(cov_syn, cov_real.detach())
    
    return loss_mean + loss_cov

def l_calib(logits_syn, labels_syn):
    return F.cross_entropy(logits_syn, labels_syn)

def l_div(logits_real, logits_syn):
    N_s = logits_syn.shape[0]
    N_r = logits_real.shape[0]
    if N_r > N_s:
        idx             = torch.randperm(N_r, device=logits_real.device)[:N_s]
        logits_real_sub = logits_real[idx].detach()
    else:
        logits_real_sub = logits_real.detach()
    p     = F.softmax(logits_real_sub, dim=-1)
    q     = F.softmax(logits_syn,      dim=-1)
    log_p = p.log().clamp(min=-100)
    log_q = q.log().clamp(min=-100)
    return (q * (log_q - log_p)).sum(dim=-1).mean()

def meancov_loss(img_real, img_syn, model, sampling_net, args):
    scale        = getattr(args, 'meancov_scale',        300.0) # default scale
    lambda_div   = getattr(args, 'hppdd_lambda_div',     0.5)
    lambda_calib = getattr(args, 'hppdd_lambda_calib',   1.0)

    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)
    logits_syn, feat_syn = model(img_syn, return_features=True)

    feat_real = F.normalize(feat_real.float(), dim=1)
    feat_syn  = F.normalize(feat_syn.float(),  dim=1)

    h_real = to_lorentz(feat_real.double())
    h_syn  = to_lorentz(feat_syn.double())

    # Mean + Covariance matching
    loss = l_meancov(h_real.detach(), h_syn) * scale

    # L_calib — semantic label consistency
    if lambda_calib > 0:
        loss = loss + lambda_calib * l_calib(logits_syn.float(),
               torch.zeros(logits_syn.shape[0], dtype=torch.long,
                           device=logits_syn.device))

    # L_div — reverse KL push
    if lambda_div > 0:
        loss = loss - lambda_div * l_div(
            logits_real.float().detach(),
            logits_syn.float()
        )

    return loss

def maybe_apply_relational(args, optim_img, model):
    pass
