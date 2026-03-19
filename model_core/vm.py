import torch
from .config import ModelConfig

class StackVM:
    def __init__(self, feat_offset=None, ops_config=None):
        self.feat_offset = feat_offset if feat_offset is not None else ModelConfig.get_feature_dim()
        _ops = ops_config if ops_config is not None else ModelConfig.get_ops_config()
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(_ops)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(_ops)}

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None