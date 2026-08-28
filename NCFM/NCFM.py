import torch
import torch.nn as nn
import torch.nn.functional as F


def ppdd_loss(img_real, img_syn, label_syn, model, args):
    """
    PPDD Unified Push-Pull Loss (ICASSP 2026)

    Per-class call: img_real and img_syn are both from the SAME class c.

    Pull forces:
      L_feat:  MSE between per-sample features (not just means)
      L_calib: Cross-entropy to preserve class semantics

    Push force:
      L_div:  Reverse KL  D_KL(q_syn || uniform)  pushes synthetic
              logit distributions toward a harder/flatter distribution,
              encouraging diversity and boundary proximity.
              (Uniform is the maximally "distant" target that does not
              require access to other-class real data at each inner step.)
    """
    model.eval()

    # ── Forward passes ─────────────────────────────────────────────────────
    with torch.no_grad():
        logits_real, feat_real = model(img_real, return_features=True)

    logits_syn, feat_syn = model(img_syn, return_features=True)

    # ── 1. PULL – Feature MSE (per-sample alignment) ───────────────────────
    # Align every synthetic feature to the mean of the real features for
    # this class.  This gives a dense gradient to each synthetic image.
    real_feat_mean = feat_real.detach().mean(dim=0, keepdim=True)   # (1, D)
    loss_mse = F.mse_loss(feat_syn, real_feat_mean.expand_as(feat_syn))

    # ── 2. PULL – Semantic calibration (Cross-Entropy) ─────────────────────
    loss_calib = F.cross_entropy(logits_syn, label_syn)

    # ── 3. PUSH – Divergence toward uniform (Reverse KL) ───────────────────
    # We maximise D_KL(q_syn || uniform) so that the synthetic images sit
    # near decision boundaries rather than in the easy centre of the class.
    #
    # D_KL(q || u) = log(C) - H(q)   where C = num_classes, H = entropy
    # Maximising D_KL  ⟺  minimising entropy  ⟺  making predictions sharp.
    #
    # But sharp AND correct is exactly what cross-entropy already enforces.
    # The *adversarial* variant instead pushes toward a WRONG-class uniform:
    #   minimise  -D_KL(q_syn || p_real)  = maximise KL(q_syn || p_real)
    # which makes the synthetic images hard wrt the CURRENT model.
    #
    # Implementation: we *subtract* KL(q_syn || p_real) from the loss so the
    # optimizer minimises (loss_mse + loss_calib) while maximising KL.
    q_syn     = F.softmax(logits_syn,         dim=1)          # (N_s, C)
    log_q_syn = F.log_softmax(logits_syn,     dim=1)          # (N_s, C)
    p_real    = F.softmax(logits_real.detach(),dim=1).mean(0) # (C,)
    log_p_real = torch.log(p_real.clamp(min=1e-8))            # (C,)

    # KL(q_syn_i || p_real)  for each synthetic sample, then average
    loss_div = (q_syn * (log_q_syn - log_p_real)).sum(dim=1).mean()

    # ── Combine ─────────────────────────────────────────────────────────────
    lambda_mse   = getattr(args, 'lambda_mse',   1.0)
    lambda_calib = getattr(args, 'lambda_calib', 1.0)
    lambda_div   = getattr(args, 'lambda_div',   0.5)

    # Pull minimises mse + calib.  Push maximises div (so we subtract it).
    total_loss = (lambda_mse * loss_mse
                  + lambda_calib * loss_calib
                  - lambda_div  * loss_div)

    return total_loss, loss_mse, loss_calib, loss_div