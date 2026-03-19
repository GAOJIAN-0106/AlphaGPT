import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        # A股模式：fdv可能为0，直接用liquidity的相对值
        if fdv.sum() < 1e-6:
            # 使用liquidity的归一化值
            liq_median = torch.nanmedian(liquidity, dim=1, keepdim=True)[0] + 1e-6
            ratio = liquidity / liq_median
            return torch.clamp(ratio, 0.0, 2.0) / 2.0
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.cat([torch.zeros_like(volume[:, :1]), volume[:, :-1]], dim=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.cat([torch.zeros_like(vol_chg[:, :1]), vol_chg[:, :-1]], dim=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window-1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1) + 1e-9))
        ret_sq = ret ** 2
        
        pad = torch.zeros((ret_sq.shape[0], window-1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)
        
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1) + 1e-9))

        pad = torch.zeros((ret.shape[0], window-1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)
        
        # Detect reversals
        mom_prev = torch.cat([torch.zeros_like(mom[:, :1]), mom[:, :-1]], dim=1)
        reversal = (mom * mom_prev < 0).float()
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1)
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        pad = torch.zeros((gains.shape[0], window-1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)
        
        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Advanced feature engineering with multiple factor types"""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Robust normalization using median absolute deviation"""
        median = torch.nanmedian(t, dim=1, keepdim=True)[0]
        mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
        norm = (t - median) / mad
        return torch.clamp(norm, -5.0, 5.0)
    
    def compute_advanced_features(self, raw_dict):
        """Compute 12-dimensional feature space with advanced factors"""
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        # Basic factors
        ret = torch.log(c / (torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev = MemeIndicators.pump_deviation(c)
        log_vol = torch.log1p(v)

        # Advanced factors
        vol_cluster = MemeIndicators.volatility_clustering(c)
        momentum_rev = MemeIndicators.momentum_reversal(c)
        rel_strength = MemeIndicators.relative_strength(c, h, l)
        
        # High-low range
        hl_range = (h - l) / (c + 1e-9)
        
        # Close position in range
        close_pos = (c - l) / (h - l + 1e-9)
        
        # Volume trend
        vol_prev = torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        
        features = torch.stack([
            self.robust_norm(ret),
            liq_score,
            pressure,
            self.robust_norm(fomo),
            self.robust_norm(dev),
            self.robust_norm(log_vol),
            self.robust_norm(vol_cluster),
            momentum_rev,
            self.robust_norm(rel_strength),
            self.robust_norm(hl_range),
            close_pos,
            self.robust_norm(vol_trend)
        ], dim=1)
        
        return features


class FeatureEngineer:
    INPUT_DIM = 12

    @staticmethod
    def compute_features(raw_dict, bars_per_day=1):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']

        # Scale rolling windows by bars_per_day so that window covers
        # the same calendar duration regardless of bar frequency.
        bpd = max(1, bars_per_day)

        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        # Basic factors (0-5)
        ret = torch.log(c / (torch.cat([torch.zeros_like(c[:, :1]), c[:, :-1]], dim=1) + 1e-9))
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v, window=5 * bpd)
        dev = MemeIndicators.pump_deviation(c, window=20 * bpd)
        log_vol = torch.log1p(v)

        # Advanced factors (6-11)
        vol_cluster = MemeIndicators.volatility_clustering(c, window=10 * bpd)
        momentum_rev = MemeIndicators.momentum_reversal(c, window=5 * bpd)
        rel_strength = MemeIndicators.relative_strength(c, h, l, window=14 * bpd)
        hl_range = (h - l) / (c + 1e-9)
        close_pos = (c - l) / (h - l + 1e-9)
        vol_prev = torch.cat([torch.zeros_like(v[:, :1]), v[:, :-1]], dim=1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)

        features = torch.stack([
            robust_norm(ret),       # 0: RET
            liq_score,              # 1: LIQ
            pressure,               # 2: PRESSURE
            robust_norm(fomo),      # 3: FOMO
            robust_norm(dev),       # 4: DEV
            robust_norm(log_vol),   # 5: LOG_VOL
            robust_norm(vol_cluster),  # 6: VOL_CLUSTER
            momentum_rev,              # 7: MOM_REV
            robust_norm(rel_strength), # 8: REL_STR
            robust_norm(hl_range),     # 9: HL_RANGE
            close_pos,                 # 10: CLOSE_POS
            robust_norm(vol_trend),    # 11: VOL_TREND
        ], dim=1)

        return features