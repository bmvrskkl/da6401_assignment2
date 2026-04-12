import torch
import torch.nn as nn


class CustomDropout(nn.Module):

    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0, "Dropout probability must be in [0, 1)"
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(x, keep_prob))
        return x * mask / keep_prob

    def extra_repr(self):
        return f"p={self.p}"
