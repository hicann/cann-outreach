import torch
import torch.nn.functional as F


def compute(inputs, attrs):
    x, dt, B, C = inputs[0], inputs[1], inputs[2], inputs[3]
    d_state = B.shape[-1]
    batch, seqlen, d_inner = x.shape

    A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1))
    D = torch.ones(d_inner, dtype=x.dtype)
    A = -torch.exp(A_log.float())

    dt = F.softplus(dt.float())
    dA = torch.exp(dt.unsqueeze(-1) * A)
    dB = dt.unsqueeze(-1) * B.float().unsqueeze(2)

    h = torch.zeros(batch, d_inner, d_state, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        h = h * dA[:, t] + dB[:, t] * x[:, t].unsqueeze(-1).float()
        C_expanded = C[:, t].unsqueeze(1).expand(-1, d_inner, -1).float()
        y = (h * C_expanded).sum(dim=-1)
        ys.append(y)
    y = torch.stack(ys, dim=1)
    y = y + D.unsqueeze(0).unsqueeze(0) * x.float()
    return y.to(x.dtype)
