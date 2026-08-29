import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference.
    Args:
        alpha_for_loss: the weight for amplitude in CF loss, from 0-1
        beta_for_loss: the weight for phase in CF loss, from 0-1
    """

    def __init__(self, alpha_for_loss=0.5, beta_for_loss=0.5):
        super(CFLossFunc, self).__init__()
        self.alpha = alpha_for_loss
        self.beta = beta_for_loss

    def forward(self, feat_tg, feat, t=None, args=None):
        """
        Calculate CF loss between target and synthetic features.
        Args:
            feat_tg: target features from real data [B1 x D]
            feat: synthetic features [B2 x D]
            args: additional arguments containing num_freqs
        """
        # Generate random frequencies
        if t is None:
            t = torch.randn((args.num_freqs, feat.size(1)), device=feat.device)
        t_x_real = calculate_real(torch.matmul(t, feat.t()))
        t_x_imag = calculate_imag(torch.matmul(t, feat.t()))
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target_real = calculate_real(torch.matmul(t, feat_tg.t()))
        t_target_imag = calculate_imag(torch.matmul(t, feat_tg.t()))
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        # Calculate amplitude difference and phase difference
        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (
            torch.mul(t_target_norm, t_x_norm)
            - torch.mul(t_x_real, t_target_real)
            - torch.mul(t_x_imag, t_target_imag)
        )

        loss_pha = loss_pha.clamp(min=1e-12)  # Ensure numerical stability

        # Combine losses
        loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss


def mmd_rbf_loss(X, Y, gamma=1.0):
    XX = torch.matmul(X, X.t())
    YY = torch.matmul(Y, Y.t())
    XY = torch.matmul(X, Y.t())
    
    rx = (X.pow(2).sum(1).unsqueeze(1).expand_as(XX))
    ry = (Y.pow(2).sum(1).unsqueeze(1).expand_as(YY))
    
    K_XX = torch.exp(-gamma * (rx.t() + rx - 2*XX))
    K_YY = torch.exp(-gamma * (ry.t() + ry - 2*YY))
    
    rx_xy = (X.pow(2).sum(1).unsqueeze(1).expand_as(XY))
    ry_xy = (Y.pow(2).sum(1).unsqueeze(0).expand_as(XY))
    K_XY = torch.exp(-gamma * (rx_xy + ry_xy - 2*XY))
    
    return K_XX.mean() + K_YY.mean() - 2*K_XY.mean()


def sinkhorn_ot_loss(x_real, x_syn, epsilon=0.05, niter=40, cost_type='sqeuclidean', debias=True):
    """
    Optimal Transport distance via Entropic Regularization (Sinkhorn Divergence).
    Matches the full geometric distribution between real and synthetic features.
    
    Args:
        x_real: Real feature embeddings [N x D] (detached)
        x_syn: Synthetic feature embeddings [M x D] (with gradients)
        epsilon: Entropic regularization parameter (controls smoothness)
        niter: Number of Sinkhorn-Knopp iterations
        cost_type: 'sqeuclidean' or 'cosine'
        debias: If True, computes the debiased Sinkhorn Divergence S_eps(x,y)
    """
    # Normalize features for well-conditioned geometry
    x_r = F.normalize(x_real.float(), dim=-1)
    x_s = F.normalize(x_syn.float(), dim=-1)

    def _compute_ot(x1, x2):
        if cost_type == 'cosine':
            C = 1.0 - torch.matmul(x1, x2.t())
        else:
            C = 2.0 - 2.0 * torch.matmul(x1, x2.t())
        C = torch.clamp(C, min=0.0)

        n1, n2 = x1.shape[0], x2.shape[0]
        mu = torch.full((n1,), 1.0 / n1, device=x1.device, dtype=x1.dtype)
        nu = torch.full((n2,), 1.0 / n2, device=x2.device, dtype=x2.dtype)
        log_mu = torch.log(mu)
        log_nu = torch.log(nu)

        f = torch.zeros(n1, device=x1.device, dtype=x1.dtype)
        g = torch.zeros(n2, device=x2.device, dtype=x2.dtype)

        for _ in range(niter):
            f = -epsilon * torch.logsumexp((-C + g.unsqueeze(0)) / epsilon, dim=1) + epsilon * log_mu
            g = -epsilon * torch.logsumexp((-C + f.unsqueeze(1)) / epsilon, dim=0) + epsilon * log_nu

        log_P = (f.unsqueeze(1) + g.unsqueeze(0) - C) / epsilon
        P = torch.exp(log_P)
        return torch.sum(P * C)

    W_xy = _compute_ot(x_r.detach(), x_s)
    if debias:
        with torch.no_grad():
            W_xx = _compute_ot(x_r.detach(), x_r.detach())
        W_yy = _compute_ot(x_s, x_s)
        return torch.clamp(W_xy - 0.5 * W_xx - 0.5 * W_yy, min=0.0)
    return W_xy


def sliced_wasserstein_loss(x_real, x_syn, num_projections=128, p=2):
    """
    Sliced Wasserstein Distance between real and synthetic feature distributions.
    """
    x_r = F.normalize(x_real.float(), dim=-1)
    x_s = F.normalize(x_syn.float(), dim=-1)
    d = x_r.shape[1]

    # Random directions on unit sphere
    projections = torch.randn(d, num_projections, device=x_r.device, dtype=x_r.dtype)
    projections = F.normalize(projections, dim=0)

    proj_x = torch.matmul(x_r, projections)
    proj_y = torch.matmul(x_s, projections)

    sort_x, _ = torch.sort(proj_x, dim=0)
    sort_y, _ = torch.sort(proj_y, dim=0)

    n, m = sort_x.shape[0], sort_y.shape[0]
    if n != m:
        # Interpolate quantiles
        grid = torch.linspace(0, n - 1, m, device=x_r.device)
        idx_low = grid.floor().long().clamp(0, n - 1)
        idx_high = grid.ceil().long().clamp(0, n - 1)
        weight = (grid - idx_low.float()).unsqueeze(1)
        sort_x = sort_x[idx_low] * (1.0 - weight) + sort_x[idx_high] * weight

    if p == 1:
        return torch.mean(torch.abs(sort_x - sort_y))
    return torch.mean((sort_x - sort_y) ** 2)


def optimal_transport_loss(feat_real, feat_syn, args):
    """Dispatcher for Optimal Transport loss based on configuration args."""
    ot_type = getattr(args, 'ot_type', 'sinkhorn').lower()
    epsilon = getattr(args, 'ot_epsilon', 0.05)
    cost_type = getattr(args, 'ot_cost', 'sqeuclidean').lower()
    scale = getattr(args, 'ot_scale', 1.0)

    if ot_type in ['sinkhorn', 'debiased_sinkhorn']:
        debias = (ot_type == 'debiased_sinkhorn') or getattr(args, 'ot_debias', True)
        return scale * sinkhorn_ot_loss(feat_real, feat_syn, epsilon=epsilon, cost_type=cost_type, debias=debias)
    elif ot_type == 'swd':
        num_proj = getattr(args, 'ot_num_projections', 128)
        return scale * sliced_wasserstein_loss(feat_real, feat_syn, num_projections=num_proj)
    else:
        # Default Sinkhorn
        return scale * sinkhorn_ot_loss(feat_real, feat_syn, epsilon=epsilon, cost_type=cost_type, debias=True)


def match_loss(img_real, img_syn, model, sampling_net, args=None, it=0):
    """Matching losses (feature Optimal Transport, MMD, or Characteristic Function)"""
    with torch.no_grad():
        _, feat_tg = model(img_real, return_features=True)
    _, feat = model(img_syn, return_features=True)
    
    metric = getattr(args, 'dis_metrics', 'OT').upper()
    if metric in ['OT', 'SINKHORN', 'WASSERSTEIN', 'SWD']:
        loss = optimal_transport_loss(feat_tg, feat, args)
    elif metric == 'MMD':
        if getattr(args, 'mmd_type', 'linear') == 'rbf':
            loss = mmd_rbf_loss(feat_tg, feat)
        else:
            loss = torch.sum((feat.mean(0) - feat_tg.mean(0)) ** 2)
    else:
        feat = F.normalize(feat, dim=1)
        feat_tg = F.normalize(feat_tg, dim=1)
        if sampling_net is not None:
            t = sampling_net(args.device)
        else:
            t = None
        loss = 300 * args.cf_loss_func(feat_tg, feat, t, args)
        
    if getattr(args, 'ppdd_lambda_div', 0.0) > 0 or getattr(args, 'ppdd_lambda_calib', 0.0) > 0:
        with torch.no_grad():
            logits_real = model(img_real, return_features=False)
        logits_syn = model(img_syn, return_features=False)
        
        # 1. Semantic Calibration (L_calib)
        c = logits_real.mean(0).argmax().item()
        target = torch.full((logits_syn.shape[0],), c, dtype=torch.long, device=logits_syn.device)
        l_calib = F.cross_entropy(logits_syn, target)
        
        # 2. Curriculum RKL (L_div)
        p_mean = F.softmax(logits_real, dim=-1).mean(dim=0)
        q_mean = F.softmax(logits_syn, dim=-1).mean(dim=0)
        log_p = p_mean.log().clamp(min=-100)
        log_q = q_mean.log().clamp(min=-100)
        rkl_loss = (q_mean * (log_q - log_p)).sum()
        
        # Apply weights
        lambda_calib = getattr(args, 'ppdd_lambda_calib', 0.0)
        lambda_div = getattr(args, 'ppdd_lambda_div', 0.0) if it >= 10000 else 0.0
        
        loss = loss + (lambda_calib * l_calib) - (lambda_div * rkl_loss)
        
    return loss


def mutil_layer_match_loss(img_real, img_syn, model,sampling_net, args=None, it=0):

    # Ensure layer_index is a list
    assert isinstance(
        args.layer_index, list
    ), "args.layer_index must be a list of layer indices"

    # Initialize loss as a tensor on the correct device
    loss = torch.tensor(0.0).to(img_real.device)

    # Extract features for both real and synthetic images
    with torch.no_grad():
        feat_tg_list = model.get_feature_mutil(img_real)  # Real image features
    feat_list = model.get_feature_mutil(img_syn)  # Synthetic image features

    for layer_index in args.layer_index:
        assert (
            0 <= layer_index <= 6
        ), f"layer_index {layer_index} must be between 0 and 6"
        metric = getattr(args, 'dis_metrics', 'OT').upper()
        if metric in ['OT', 'SINKHORN', 'WASSERSTEIN', 'SWD']:
            loss += optimal_transport_loss(feat_tg_list[layer_index], feat_list[layer_index], args)
        elif metric == "MMD":
            # If the metric is MMD, calculate the MMD loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            loss += torch.sum((feat.mean(0) - feat_tg.mean(0)) ** 2)
        else:
            # Otherwise, calculate the feature matching loss for the selected layer
            feat = feat_list[layer_index]
            feat_tg = feat_tg_list[layer_index]
            feat = F.normalize(feat, dim=1)  # Normalize the feature
            feat_tg = F.normalize(feat_tg, dim=1)  # Normalize the target feature
            t = None  # Adjust this based on your CFLossFunc usage
            loss += 300 * args.cf_loss_func(feat_tg, feat, t, args)

    if getattr(args, 'ppdd_lambda_div', 0.0) > 0 or getattr(args, 'ppdd_lambda_calib', 0.0) > 0:
        with torch.no_grad():
            logits_real = model(img_real, return_features=False)
        logits_syn = model(img_syn, return_features=False)
        
        # 1. Semantic Calibration (L_calib)
        c = logits_real.mean(0).argmax().item()
        target = torch.full((logits_syn.shape[0],), c, dtype=torch.long, device=logits_syn.device)
        l_calib = F.cross_entropy(logits_syn, target)
        
        # 2. Curriculum RKL (L_div)
        p_mean = F.softmax(logits_real, dim=-1).mean(dim=0)
        q_mean = F.softmax(logits_syn, dim=-1).mean(dim=0)
        log_p = p_mean.log().clamp(min=-100)
        log_q = q_mean.log().clamp(min=-100)
        rkl_loss = (q_mean * (log_q - log_p)).sum()
        
        # Apply weights
        lambda_calib = getattr(args, 'ppdd_lambda_calib', 0.0)
        lambda_div = getattr(args, 'ppdd_lambda_div', 0.0) if it >= 10000 else 0.0
        
        loss = loss + (lambda_calib * l_calib) - (lambda_div * rkl_loss)

    return loss


def cailb_loss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)
    return loss