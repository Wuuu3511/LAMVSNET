import torch
import torch.nn.functional as F


def mvs_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", [1.0 for k in inputs.keys() if "stage" in k])
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]  # (b, d, h, w)
        depth_est = stage_inputs["depth"]  # (b, h, w)
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        interval = stage_inputs["interval"]  # float
        depth_gt = depth_gt_ms[stage_key]  # (b, h, w)
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        stage_idx = int(stage_key.replace("stage", "")) - 1
        stage_weight = depth_loss_weights[stage_idx]

    
        loss = classification_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight)
        total_loss += loss

    return total_loss


def classification_loss(prob_volume, depth_values, interval, depth_gt, mask, weight):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (
            ((depth_values - interval / 2) <= depth_gt_volume).float() * ((depth_values + interval / 2) > depth_gt_volume).float())

    NEAR_0 = 1e-4  # Prevent overflow
    prob_volume = torch.where(prob_volume <= 0.0, torch.zeros_like(prob_volume) + NEAR_0, prob_volume)

    loss = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1)[mask].mean()
    loss = loss * weight
    return loss

