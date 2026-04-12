import torch
import torch.nn as nn


class IoULoss(nn.Module):

    def __init__(self, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        assert reduction in ("mean", "sum"), \
            f"reduction must be 'mean' or 'sum', got '{reduction}'"
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : [B, 4]  predicted boxes  [cx, cy, w, h] in pixel space
            target : [B, 4]  ground truth boxes [cx, cy, w, h] in pixel space

        Returns:
            Scalar loss in range [0, 1].
        """
        # Convert cx,cy,w,h -> x1,y1,x2,y2
        def to_xyxy(box):
            x1 = box[:, 0] - box[:, 2] / 2
            y1 = box[:, 1] - box[:, 3] / 2
            x2 = box[:, 0] + box[:, 2] / 2
            y2 = box[:, 1] + box[:, 3] / 2
            return x1, y1, x2, y2

        px1, py1, px2, py2 = to_xyxy(pred)
        tx1, ty1, tx2, ty2 = to_xyxy(target)

        # Intersection
        ix1 = torch.max(px1, tx1)
        iy1 = torch.max(py1, ty1)
        ix2 = torch.min(px2, tx2)
        iy2 = torch.min(py2, ty2)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        # Union
        area_pred   = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_target = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = area_pred + area_target - inter + self.eps

        iou  = inter / union                  # [B], range [0,1]
        loss = 1.0 - iou                      # [B], range [0,1]

        if self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()

    def extra_repr(self):
        return f"reduction={self.reduction}"
