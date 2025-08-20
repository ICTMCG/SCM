import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicConsistencyLoss(nn.Module):

    def __init__(self, n_classes, reduce="max", reduction="mean"):
        super(LogicConsistencyLoss, self).__init__()
        if reduction not in ["sum", "mean", "none"]:
            raise ValueError(
                f"reduction should be 'sum', 'mean', or 'none', but got {reduction}"
            )
        if reduce not in ["max", "min", "avg"]:
            raise ValueError(
                f"reduce should be 'max', 'avg' or 'min', but got {reduce}"
            )
        self.n_classes = n_classes
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, fc, gc, gc_mask):
        batch_num, token_num, class_num = gc.size()
        assert (
            class_num == self.n_classes
        ), f"Class number mismatch: {class_num} vs {self.n_classes}"
        fc = F.softmax(fc, dim=-1)[..., -1]
        gc = F.softmax(gc, dim=-1)[..., -1]
        gc = torch.mul(gc, gc_mask)

        if self.reduce == "max":
            gc_maxpool, _ = torch.max(gc, dim=-1)
        elif self.reduce == "min":
            gc_maxpool, _ = torch.min(gc, dim=-1)
        elif self.reduce == "avg":
            gc_maxpool = torch.mean(gc, dim=-1)
        pf = 1 - fc + torch.mul(fc, gc_maxpool)
        pf = torch.clamp(pf, min=1e-8)
        out = torch.neg(torch.log(pf))

        if self.reduction == "mean":
            loss = torch.mean(out)
        elif self.reduction == "sum":
            loss = torch.sum(out)
        return loss
