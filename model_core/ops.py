import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]


# ---------------------------------------------------------------------------
# Extended operators for futures (intraday microstructure search space)
# ---------------------------------------------------------------------------

def _op_cs_rank(x: torch.Tensor) -> torch.Tensor:
    """Cross-sectional rank normalised to [-1, 1].  dim=0 = across products."""
    n = x.shape[0]
    if n <= 1:
        return torch.zeros_like(x)
    ranks = x.argsort(dim=0).argsort(dim=0).float()
    return ranks / (n - 1) * 2.0 - 1.0

def _op_cs_demean(x: torch.Tensor) -> torch.Tensor:
    """Remove cross-sectional mean (market factor).  dim=0 = across products."""
    return x - x.mean(dim=0, keepdim=True)

def _op_ma(x: torch.Tensor, window: int) -> torch.Tensor:
    """Simple moving average along time (dim=1), zero-padded."""
    if x.shape[1] < window:
        return x
    pad = torch.zeros((x.shape[0], window - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    return x_pad.unfold(1, window, 1).mean(dim=-1)

def _op_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    """x[t] - x[t-d]."""
    return x - _ts_delay(x, d)

def _op_ts_std(x: torch.Tensor, window: int) -> torch.Tensor:
    """Rolling standard deviation along time (dim=1)."""
    if x.shape[1] < window:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], window - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    unf = x_pad.unfold(1, window, 1)
    return unf.std(dim=-1)

def _op_corr10(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """10-bar rolling Pearson correlation. Output clamped to [-1, 1]."""
    window = 10
    if x.shape[1] < window:
        return torch.zeros_like(x)
    pad = torch.zeros((x.shape[0], window - 1), device=x.device)
    x_pad = torch.cat([pad, x], dim=1)
    y_pad = torch.cat([pad, y], dim=1)
    x_unf = x_pad.unfold(1, window, 1)
    y_unf = y_pad.unfold(1, window, 1)
    x_m = x_unf - x_unf.mean(dim=-1, keepdim=True)
    y_m = y_unf - y_unf.mean(dim=-1, keepdim=True)
    cov = (x_m * y_m).mean(dim=-1)
    std_x = x_unf.std(dim=-1) + 1e-8
    std_y = y_unf.std(dim=-1) + 1e-8
    corr = cov / (std_x * std_y)
    return torch.clamp(corr, -1.0, 1.0)

def _op_ifelse(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Soft if-else: sigmoid(condition) * x + (1 - sigmoid(condition)) * y.
    Unlike GATE (hard threshold on condition>0), this uses a smooth sigmoid
    to allow gradient flow through both branches.
    """
    w = torch.sigmoid(condition)
    return w * x + (1.0 - w) * y

def _op_cond_mul(condition: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Conditional multiply: sigmoid(condition) * x.
    Arity-2 alternative to GATE for conditional scaling.
    """
    return torch.sigmoid(condition) * x

OPS_CONFIG_EXTENDED = OPS_CONFIG + [
    ('CS_RANK',   _op_cs_rank,                     1),  # 12
    ('CS_DEMEAN', _op_cs_demean,                    1),  # 13
    ('MA5',       lambda x: _op_ma(x, 5),          1),  # 14
    ('MA20',      lambda x: _op_ma(x, 20),         1),  # 15
    ('DELTA5',    lambda x: _op_delta(x, 5),       1),  # 16
    ('TS_STD10',  lambda x: _op_ts_std(x, 10),     1),  # 17
    ('DELAY5',    lambda x: _ts_delay(x, 5),       1),  # 18
    ('CORR10',    _op_corr10,                       2),  # 19
    ('IFELSE',    _op_ifelse,                       3),  # 20
    ('COND_MUL',  _op_cond_mul,                     2),  # 21
]