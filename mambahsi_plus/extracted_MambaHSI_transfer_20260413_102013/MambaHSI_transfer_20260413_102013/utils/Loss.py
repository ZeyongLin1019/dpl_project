import warnings
import torch
import torch.nn.functional as F


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def head_loss(loss_func,logits,label,align_corners=True):
    seg_logits = resize(
        input=logits,
        size=label.shape[1:],
        mode='bilinear',
        align_corners=align_corners)

    loss = loss_func(seg_logits,label)
    return loss


def spectral_reconstruction_loss(recon, x, target, lambda_sam=0.5, lambda_l2=0.03, eps=1e-6):
    if recon.shape[2:] != x.shape[2:]:
        recon = resize(
            input=recon,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False,
            warning=False,
        )

    if target.dim() == 2:
        target = target.unsqueeze(0)
    if target.dim() == 3:
        mask = (target != -1).unsqueeze(1).float()
    else:
        raise ValueError('target should have shape [H,W] or [B,H,W]')

    mask = mask.to(recon.device)
    valid = mask.sum()
    if valid.item() == 0:
        return recon.sum() * 0.0

    diff = recon - x

    mse_map = diff.pow(2).mean(dim=1, keepdim=True)
    mse_loss = (mse_map * mask).sum() / (valid + eps)

    l2_map = torch.sqrt(diff.pow(2).sum(dim=1, keepdim=True) + eps)
    l2_loss = (l2_map * mask).sum() / (valid + eps)

    cosine = F.cosine_similarity(recon, x, dim=1, eps=eps).unsqueeze(1)
    sam_map = 1.0 - cosine
    sam_loss = (sam_map * mask).sum() / (valid + eps)

    loss = mse_loss + float(lambda_sam) * sam_loss + float(lambda_l2) * l2_loss
    if not torch.isfinite(loss):
        return recon.sum() * 0.0
    return loss