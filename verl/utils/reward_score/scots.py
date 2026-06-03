import torch
from scipy.optimize import linear_sum_assignment

from verl.utils.reward_score.rseg_cot import format_reward
from verl.utils.torch_functional import pairwise_soft_iou


def scots_compute_score(
    tokens: str,
    masks: torch.Tensor = None,
    gt_masks: torch.Tensor = None,
    frame_logits: torch.Tensor = None,
    frame_labels: torch.Tensor = None,
    n_multi_objects: torch.Tensor = None,
    conf_logits: torch.Tensor = None,
):
    """
    Compute scalar score = 0.3*format_reward(tokens) + 0.7*mean mask IoU.

    Args:
        tokens: prompt/response text.
        masks: predicted mask probabilities, shape [K, H, W].
        gt_masks: ground-truth masks, shape [G, H, W], padded to K for multi-object data.
        n_multi_objects: scalar tensor; -1 for single-object data, otherwise number of valid gt masks.
        conf_logits: kept for call-site compatibility; reward uses all K predicted slots.
    """
    format_score = format_reward(tokens)
    if frame_logits is not None and frame_labels is not None:
        score = _frame_mean_iou(frame_logits, frame_labels)
    else:
        score = _slot_mean_iou(masks, gt_masks, n_multi_objects)
    return 0.3 * float(format_score) + 0.7 * score


def _frame_mean_iou(frame_logits: torch.Tensor, frame_labels: torch.Tensor) -> float:
    logits = torch.as_tensor(frame_logits, dtype=torch.float32)
    if logits.dim() == 3 and logits.size(1) == 1:
        logits = logits[:, 0, :]
    elif logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() != 2:
        raise ValueError(f"frame_logits must be [K,N] or [K,1,N], got {tuple(logits.shape)}")
    pred = torch.sigmoid(logits)

    labels = torch.as_tensor(frame_labels, device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)
    K, N = pred.shape
    if labels.dim() == 1:
        if labels.numel() != N:
            labels = torch.nn.functional.interpolate(
                labels.view(1, 1, -1), size=N, mode="nearest"
            ).view(N)
        labels = labels.view(1, N).expand(K, N)
    elif labels.dim() == 2 and labels.shape == (1, N):
        labels = labels.expand(K, N)
    elif labels.dim() != 2 or labels.shape != (K, N):
        raise ValueError(f"cannot align frame_labels shape {tuple(labels.shape)} to pred {(K, N)}")

    intersection = (pred * labels).sum(dim=-1)
    union = pred.sum(dim=-1) + labels.sum(dim=-1) - intersection
    iou = intersection / union.clamp_min(1e-6)
    return float(iou.mean().item())


def _slot_mean_iou(masks: torch.Tensor, gt_masks: torch.Tensor, n_multi_objects: torch.Tensor | None) -> float:
    pred = torch.as_tensor(masks, dtype=torch.float32).clamp(0.0, 1.0)
    gt = torch.as_tensor(gt_masks, device=pred.device, dtype=pred.dtype).clamp(0.0, 1.0)

    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred[:, 0]
    if gt.dim() == 4 and gt.size(1) == 1:
        gt = gt[:, 0]
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0)
    if pred.dim() != 3 or gt.dim() != 3:
        raise ValueError(f"masks and gt_masks must be [K,H,W], got {tuple(pred.shape)} and {tuple(gt.shape)}")

    if n_multi_objects is None:
        n_gt = gt.shape[0]
    else:
        n_gt = int(torch.as_tensor(n_multi_objects).item())
        if n_gt < 0:
            n_gt = gt.shape[0]
    gt = gt[:n_gt]

    K, G = pred.shape[0], gt.shape[0]
    if K == 0 and G == 0:
        return 1.0
    if K == 0 or G == 0:
        return 0.0

    iou = pairwise_soft_iou(pred, gt)
    m = max(K, G)
    iou_pad = iou.new_zeros((m, m))
    iou_pad[:K, :G] = iou
    rows, cols = linear_sum_assignment((1.0 - iou_pad).detach().cpu().numpy())
    return float(iou_pad[rows, cols].mean().item())
