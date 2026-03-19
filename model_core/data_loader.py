import pandas as pd
import torch
import sqlalchemy
from .config import ModelConfig, TimeframeProfile
from .factors import FeatureEngineer

class CryptoDataLoader:
    def __init__(self, train_ratio=0.7, timeframe: str = None):
        self.engine = sqlalchemy.create_engine(ModelConfig.DB_URL)
        self.train_ratio = train_ratio
        self.timeframe = timeframe or ModelConfig.TIMEFRAME
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        
    def load_data(self, limit_tokens=500):
        print("Loading data from SQL...")
        top_query = f"""
        SELECT address FROM tokens 
        LIMIT {limit_tokens} 
        """
        addrs = pd.read_sql(top_query, self.engine)['address'].tolist()
        if not addrs: raise ValueError("No tokens found.")
        addr_str = "'" + "','".join(addrs) + "'"
        data_query = f"""
        SELECT time, address, open, high, low, close, volume, liquidity, fdv
        FROM ohlcv
        WHERE address IN ({addr_str})
          AND timeframe = '{self.timeframe}'
        ORDER BY time ASC
        """
        df = pd.read_sql(data_query, self.engine)
        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv')
        }
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # 保存日期索引（用于年度分析）
        self.dates = sorted(df['time'].unique().tolist())

        # 计算目标收益率 (次日收益)
        close = self.raw_data_cache['close']
        next_close = torch.cat([close[:, 1:], torch.zeros_like(close[:, :1])], dim=1)
        self.target_ret = (next_close - close) / (close + 1e-9)
        self.target_ret[:, -1] = 0.0  # 最后一天无收益

        # 清理异常值（clamp 随周期自动缩放：日线±20%，15min±5%，5min±2.9%…）
        profile = TimeframeProfile(self.timeframe, ModelConfig.ASSET_CLASS)
        self.target_ret = torch.clamp(self.target_ret, -profile.ret_clamp, profile.ret_clamp)
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0)

        print(f"Data Ready. Shape: {self.feat_tensor.shape}")

        # Temporal train/test split
        time_steps = self.feat_tensor.shape[2]
        split_idx = int(time_steps * self.train_ratio)

        # Feature split [num_tokens, num_features, time_steps]
        self.train_feat = self.feat_tensor[:, :, :split_idx]
        self.test_feat = self.feat_tensor[:, :, split_idx:]

        # Target return split [num_tokens, time_steps]
        self.train_ret = self.target_ret[:, :split_idx]
        self.test_ret = self.target_ret[:, split_idx:]

        # Raw data cache split [num_tokens, time_steps]
        self.train_raw = {k: v[:, :split_idx] for k, v in self.raw_data_cache.items()}
        self.test_raw = {k: v[:, split_idx:] for k, v in self.raw_data_cache.items()}

        print(f"Split: train={split_idx} steps, test={time_steps - split_idx} steps")