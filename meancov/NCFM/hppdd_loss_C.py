# H-PPDD Option C: Learnable interpolation between Euclidean and Hyperbolic
# A scalar alpha (learned) blends Euclidean MSE and hyperbolic centroid loss.
# alpha=0 -> pure Euclidean MSE (like PPDD)
# alpha=1 -> pure hyperbolic centroid (like HDD)
# The model learns the optimal geometry mix during distillation.
# Loss = scale * (alpha * L_hyp + (1-alpha) * L_euc) - lambda_div * L_div

import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.lorentz import math as lmath

K_CURV = 1
man    = Lorentz(k=K_CURV)

# Learnable alpha — initialised at 0.5 (equal blend)
_alpha = nn.Parameter(torch.tensor(0.5))

def to_lorentz(feat, eps=1e-8):
    h = man.expmap0(F.pad(feat, pad=(1, 0)))
    return man.projx(h)

def hdd_centroid(x, eps=1e-8):
    k     = torch.tensor(float(K_CURV), device=x.device, dtype=x.dtype)
    avg   = x.mean(dim=-2)
    denom = (-lmath.inner(avg, avg, keepdim=True)).abs().clamp_min(eps).sqrt()
    return torch.sqrt(k) * avg / denom

def l_hyp_centroid(h_real, h_syn):
    cr = hdd_centroid(h_real).detach()
    cs = hdd_centroid(h_syn)
    return man.dist(cr.unsqueeze(0), cs.unsqueeze(0))

def l_euc_mse(f_real, f_syn):
    mean_real = f_real.detach().mean(dim=0)
    mean_syn  = f_syn.mean(dim=0)
    return F.mse_loss(mean_syn, mean_real)

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

def hppdd_loss(img_real, img_syn, model, sampling_net, args):
    global _alpha

    scale      = getattr(args, 'hppdd_scale',      300.0)
    lambda_div = getattr(args, 'hppdd_lambda_div',  0.5)

    # Move alpha to same device as data
    if _alpha.device != torch.device(args.device):
        _alpha = _alpha.to(args.device)

    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)
    logits_syn, feat_syn = model(img_syn, return_features=True)

    feat_real = F.normalize(feat_real.float(), dim=1)
    feat_syn  = F.normalize(feat_syn.float(),  dim=1)

    h_real = to_lorentz(feat_real.double())
    h_syn  = to_lorentz(feat_syn.double())

    # Clamp alpha to [0, 1]
    alpha = _alpha.clamp(0.0, 1.0)

    l_hyp = l_hyp_centroid(h_real, h_syn)
    l_euc = l_euc_mse(feat_real, feat_syn)

    # Normalise both to similar scale before blending
    loss = scale * (alpha * l_hyp + (1.0 - alpha) * l_euc * 10.0)

    if lambda_div > 0:
        loss = loss - lambda_div * l_div(
            logits_real.float().detach(),
            logits_syn.float()
        )

    # Log alpha every 100 iters
    iter_id = getattr(args, 'hppdd_iter', 0)
    if iter_id % 100 == 0 and iter_id > 0:
        print(f'[H-PPDD-C] iter={iter_id} alpha={alpha.item():.4f}')

    return loss

def maybe_apply_relational(args, optim_img, model):
    pass
