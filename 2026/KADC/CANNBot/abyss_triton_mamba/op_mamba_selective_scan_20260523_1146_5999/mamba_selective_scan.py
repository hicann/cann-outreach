import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, d_inner, d_state):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x, dt, B, C):
        batch, seqlen, d_inner = x.shape
        d_state = self.d_state

        A_log_device = self.A_log.to(x.device)
        D_device = self.D.to(x.device)
        A = -torch.exp(A_log_device.float())

        dt = F.softplus(dt)

        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch, d_inner, d_state, dtype=x.dtype, device=x.device)
        ys = []

        for t in range(seqlen):
            h = h * dA[:, t] + dB[:, t] * x[:, t].unsqueeze(-1)
            C_expanded = C[:, t].unsqueeze(1).expand(-1, d_inner, -1)
            y = (h * C_expanded).sum(dim=-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + D_device.unsqueeze(0).unsqueeze(0) * x

        return y


def get_init_inputs():
    return [32, 8]


def get_input_groups():
    d_inner = 32
    d_state = 8
    groups = []
    configs = [
        (2, 128),
        (2, 256),
        (4, 128),
        (4, 256),
        (2, 512),
    ]
    for batch, seqlen in configs:
        x = torch.randn(batch, seqlen, d_inner, dtype=torch.float32)
        dt = torch.randn(batch, seqlen, d_inner, dtype=torch.float32)
        B = torch.randn(batch, seqlen, d_state, dtype=torch.float32)
        C = torch.randn(batch, seqlen, d_state, dtype=torch.float32)
        groups.append([x, dt, B, C])
    return groups