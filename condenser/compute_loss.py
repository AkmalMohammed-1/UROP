import torch

def compute_match_loss(
    args,
    loader_real,
    sample_fn,
    aug_fn,
    inner_loss_fn,
    optim_img,
    class_list,
    timing_tracker,
    model_interval,
    data_grad,
    optim_sampling_net = None,
    sampling_net =None
):

    loss_total = 0
    match_grad_mean = 0

    for c in class_list:
        timing_tracker.start_step()

        img, _ = loader_real.class_sample(c)
        timing_tracker.record("data")
        img_syn, label_syn = sample_fn(c)

        img_aug = aug_fn(torch.cat([img, img_syn]))
        timing_tracker.record("aug")
        n = img.shape[0]

        loss, loss_mse, loss_calib, loss_div = inner_loss_fn(img_aug[:n], img_aug[n:], label_syn, model_interval, args)
        loss_total += loss.item()
        timing_tracker.record("loss")

        optim_img.zero_grad()
        loss.backward()
        optim_img.step()
        if data_grad is not None:
            match_grad_mean += torch.norm(data_grad).item()
        timing_tracker.record("backward")
        
        if hasattr(args, 'use_wandb') and args.use_wandb and getattr(args, 'rank', 0) == 0:
            import wandb
            wandb.log({
                'loss/total': loss.item(),
                'loss/mse': loss_mse.item(),
                'loss/calib': loss_calib.item(),
                'loss/div': loss_div.item(),
            })

    return loss_total, match_grad_mean

def compute_calib_loss(*args, **kwargs):
    # PPDD combines calib loss into the main update, so this is unused.
    return 0, 0

