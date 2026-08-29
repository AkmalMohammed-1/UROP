import torch
import torch.nn as nn
import torch.nn.functional as F


def ppdd_loss(img_real, img_syn, label_syn, model, args):
    """
    PPDD Unified Push-Pull Loss (ICASSP 2026)

    Per-class call: img_real and img_syn are both from the SAME class c.

    Pull forces (Eq 5):
      L_MSE:   MSE between per-sample features and the real feature mean (Eq 3)
      L_calib: Cross-entropy for label consistency (Eq 4)

    Push force (Eq 6):
      L_div:   Reverse KL  D_KL(q(y|x_s) || p(y|x_r))  per-sample paired
    """
    # ── Forward passes ─────────────────────────────────────────────────────
    # We do NOT use model.eval() because it disrupts instance normalization.
    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)

    logits_syn, feat_syn = model(img_syn, return_features=True)

    # ── 1. PULL – Feature MSE (Eq 3) ──────────────────────────────────────
    # IMPORTANT: We must match EVERY synthetic feature to the real mean.
    # This implicitly penalizes the variance of the synthetic features.
    # Matching mean-to-mean without normalizing the features causes the 
    # synthetic variance to explode due to the push loss!
    real_feat_mean = feat_real.detach().mean(dim=0, keepdim=True)
    loss_mse = F.mse_loss(feat_syn, real_feat_mean.expand_as(feat_syn))

    # ── 2. PULL – Semantic calibration (Eq 4) ─────────────────────────────
    loss_calib = F.cross_entropy(logits_syn, label_syn)

    # ── 3. PUSH – Reverse KL divergence (Eq 6) ────────────────────────────
    # Per-sample pairing: subsample real to match synthetic batch size,
    # then compute KL(q_i || p_i) for each pair and average.
    loss_div = _l_div(logits_real.detach(), logits_syn)

    # ── Combine (Eq 2): L_PPDD = L_align - λ_div * L_div ─────────────────
    lambda_mse   = getattr(args, 'lambda_mse',   1.0)
    lambda_calib = getattr(args, 'lambda_calib', 1.0)
    lambda_div   = getattr(args, 'lambda_div',   0.5)

    loss_align = lambda_mse * loss_mse + lambda_calib * loss_calib
    total_loss = loss_align - lambda_div * loss_div

    return total_loss, loss_mse, loss_calib, loss_div


def _l_div(logits_real, logits_syn):
    """
    Per-sample paired Reverse KL: D_KL(q(y|x_s) || p(y|x_r))

    Subsamples real to match synthetic batch size so every synthetic
    sample is paired with a distinct real sample.
    """
    N_s = logits_syn.shape[0]
    N_r = logits_real.shape[0]

    if N_r > N_s:
        idx = torch.randperm(N_r, device=logits_real.device)[:N_s]
        logits_real_sub = logits_real[idx]
    else:
        logits_real_sub = logits_real

    p     = F.softmax(logits_real_sub, dim=-1)
    q     = F.softmax(logits_syn,      dim=-1)
    log_p = p.log().clamp(min=-100)
    log_q = q.log().clamp(min=-100)

    return (q * (log_q - log_p)).sum(dim=-1).mean()