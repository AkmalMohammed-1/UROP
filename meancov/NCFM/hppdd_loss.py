# H-MMD: Hyperbolic Maximum Mean Discrepancy
# Replaces NCFM's characteristic function with a kernel-based distribution
# matching loss on the Lorentz manifold.
#
# L_HMMD = E[k(hr,hr')] - 2*E[k(hr,hs)] + E[k(hs,hs')]
# where k(x,y) = exp(-d_H(x,y)^2 / sigma^2)  [hyperbolic RBF kernel]
#
# This implicitly matches ALL moments of the hyperbolic feature distribution
# unlike HDD which only matches the 1st moment (centroid).
# Mathematically equivalent to NCFM's CF matching but in hyperbolic space.
#
# Loss = scale * L_HMMD - lambda_div * L_div + lambda_calib * L_calib

import torch
import torch.nn.functional as F
from geoopt.manifolds.lorentz import Lorentz
from geoopt.manifolds.lorentz import math as lmath

K_CURV = 1
man    = Lorentz(k=K_CURV)

def to_lorentz(feat, eps=1e-8):
    h = man.expmap0(F.pad(feat, pad=(1, 0)))
    return man.projx(h)

def hyp_rbf_kernel(x, y, sigma=1.0):
    """
    Hyperbolic RBF kernel: k(x,y) = exp(-d_H(x,y)^2 / sigma^2)
    x : (N, D+1) Lorentz vectors
    y : (M, D+1) Lorentz vectors
    returns : (N, M) kernel matrix
    """
    N = x.shape[0]
    M = y.shape[0]
    K = torch.zeros(N, M, device=x.device, dtype=x.dtype)
    for i in range(N):
        d = man.dist(x[i].unsqueeze(0).expand(M, -1), y)  # (M,)
        K[i] = torch.exp(-d.pow(2) / (sigma ** 2))
    return K

def l_hmmd(h_real, h_syn, sigma=1.0):
    """
    Hyperbolic MMD between real and synthetic hyperbolic embeddings.
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

    # k(real, real) — fully detached
    with torch.no_grad():
        K_rr = hyp_rbf_kernel(h_real, h_real, sigma)

    # k(syn, syn) — has gradient through h_syn
    K_ss = hyp_rbf_kernel(h_syn, h_syn, sigma)

    # k(real, syn) — has gradient through h_syn
    K_rs = hyp_rbf_kernel(h_real.detach(), h_syn, sigma)

    mmd = (K_rr.sum() / (N_r * N_r)
           - 2 * K_rs.sum() / (N_r * N_s)
           + K_ss.sum() / (N_s * N_s))

    return mmd

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

_label_buf = {}
_current_iter = -1

def hppdd_loss(img_real, img_syn, model, sampling_net, args):
    global _label_buf, _current_iter

    scale        = getattr(args, 'hppdd_scale',        300.0)
    lambda_div   = getattr(args, 'hppdd_lambda_div',    0.5)
    lambda_calib = getattr(args, 'hppdd_lambda_calib',  1.0)
    sigma        = getattr(args, 'hppdd_sigma',         1.0)

    iter_id = getattr(args, 'hppdd_iter', 0)
    if iter_id != _current_iter:
        _label_buf    = {}
        _current_iter = iter_id

    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)
    logits_syn, feat_syn = model(img_syn, return_features=True)

    feat_real = F.normalize(feat_real.float(), dim=1)
    feat_syn  = F.normalize(feat_syn.float(),  dim=1)

    h_real = to_lorentz(feat_real.double())
    h_syn  = to_lorentz(feat_syn.double())

    # L_HMMD — full distribution matching in hyperbolic space
    loss = l_hmmd(h_real.detach(), h_syn, sigma=sigma) * scale

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
