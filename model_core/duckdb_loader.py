"""
Direct DuckDB → Tensor data loader.

Bypasses PostgreSQL, loads main-contract commodity futures data directly
from the local DuckDB file into GPU tensors. Supports any timeframe via
DuckDB-side aggregation.

Term-structure support: also loads the 2nd-ranked contract (by OI) to
compute F1/F2 spreads, Pindyck volatility, and carry factors.
"""

import re
import pandas as pd
import torch
from .config import ModelConfig, TimeframeProfile
from .factors import FeatureEngineer
from .features_v2 import DuckDBFeatureEngineer

try:
    import duckdb
except ImportError:
    duckdb = None

# Timeframe → minutes for DuckDB aggregation
_TF_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}


class DuckDBDataLoader:
    """Load commodity futures data directly from DuckDB into tensors."""

    def __init__(self, train_ratio=0.7, timeframe: str = None,
                 db_path: str = None, products: list = None,
                 end_date: str = None, skip_ic_screening: bool = False):
        if duckdb is None:
            raise ImportError("duckdb is required. Run: pip install duckdb")

        self.train_ratio = train_ratio
        self.timeframe = timeframe or ModelConfig.TIMEFRAME
        self.db_path = db_path or self._default_db_path()
        self.products = products  # None = all
        self.end_date = end_date  # None = all, 'YYYY-MM-DD' = up to date
        self.skip_ic_screening = skip_ic_screening  # True = keep all features

        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None
        self.dates = None

    @staticmethod
    def _default_db_path():
        import os
        return os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))

    def load_data(self, limit_tokens=500):
        minutes = _TF_MINUTES.get(self.timeframe, 1440)
        con = duckdb.connect(self.db_path, read_only=True)

        self._term_con = None  # hold connection for deferred term structure query
        try:
            df = self._query_main_contracts(con, minutes, limit_tokens)
            # Query secondary contracts now (while connection is open), compute later
            self._secondary_df = None
            self._has_term_structure = False
            self._has_tide = False
            if minutes >= 1440:
                try:
                    self._secondary_df = self._query_secondary_contracts(con)
                    if self._secondary_df is not None and not self._secondary_df.empty:
                        self._has_term_structure = True
                        print(f"Secondary contracts: {len(self._secondary_df)} rows loaded")
                except Exception as e:
                    print(f"Warning: secondary contract query failed: {e}")
                # Tide factor needs connection — store con reference for later
                self._tide_con_path = self.db_path
        finally:
            con.close()

        if df.empty:
            raise ValueError("No data returned from DuckDB")

        print(f"DuckDB loaded: {len(df):,} rows, "
              f"{df['address'].nunique()} products, "
              f"timeframe={self.timeframe}")

        device = ModelConfig.DEVICE

        def to_tensor(col):
            pivot = df.pivot(index='time', columns='address', values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=device)

        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'liquidity': to_tensor('liquidity'),
            'fdv': to_tensor('fdv'),
        }

        # Detect intraday columns and compute derived microstructure features
        has_intraday = 'vwap' in df.columns and 'twap' in df.columns
        if has_intraday:
            close_t = self.raw_data_cache['close']
            vwap_t = to_tensor('vwap')
            oi_change_t = to_tensor('oi_change')
            am_vol_t = to_tensor('am_volume')
            pm_vol_t = to_tensor('pm_volume')
            max_30min_t = to_tensor('max_30min_vol')
            volume_t = self.raw_data_cache['volume']
            last_hour_t = to_tensor('last_hour_vol')
            first_hour_t = to_tensor('first_hour_vol')
            twap_t = to_tensor('twap')

            self.raw_data_cache['vwap_dev'] = (vwap_t - close_t) / (close_t + 1e-9)
            self.raw_data_cache['oi_change'] = oi_change_t
            if 'total_oi' in df.columns:
                self.raw_data_cache['total_oi'] = to_tensor('total_oi')
            self.raw_data_cache['vol_skew'] = am_vol_t / (pm_vol_t + 1e-9)
            self.raw_data_cache['vol_conc'] = max_30min_t / (volume_t + 1e-9)
            self.raw_data_cache['smart_money'] = last_hour_t / (first_hour_t + 1e-9)
            self.raw_data_cache['twap_dev'] = (twap_t - close_t) / (close_t + 1e-9)

            # Volume Support Price position (from SQL-side computation)
            if 'vsp_position' in df.columns:
                self.raw_data_cache['vsp_position'] = to_tensor('vsp_position')
            # Buy willingness ratio (from SQL-side computation)
            if 'buy_ratio' in df.columns:
                self.raw_data_cache['buy_ratio'] = to_tensor('buy_ratio')
            # Large order impact (from SQL-side computation)
            if 'large_order_ret' in df.columns:
                self.raw_data_cache['large_order_ret'] = to_tensor('large_order_ret')
            # Intraday RSI from 1min data
            if 'intraday_rsi' in df.columns:
                self.raw_data_cache['intraday_rsi'] = to_tensor('intraday_rsi')
            # Volume ratio from 1min data
            if 'vol_ratio' in df.columns:
                self.raw_data_cache['vol_ratio'] = to_tensor('vol_ratio')
            # Inflection point ratio from 1min data
            if 'inflection_ratio' in df.columns:
                self.raw_data_cache['inflection_ratio'] = to_tensor('inflection_ratio')
            # Flow in ratio from 1min data
            if 'flow_in_ratio' in df.columns:
                self.raw_data_cache['flow_in_ratio'] = to_tensor('flow_in_ratio')

            self.feat_tensor = DuckDBFeatureEngineer.compute_features(self.raw_data_cache)
        else:
            from .config import TIMEFRAME_BARS_PER_DAY
            bpd = TIMEFRAME_BARS_PER_DAY.get(self.timeframe, 1)
            self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache, bars_per_day=bpd)

        # Compute tide factor — use cached parquet if available, else compute
        if hasattr(self, '_tide_con_path') and not self._has_tide:
            import os
            cache_path = os.path.join(os.path.dirname(self._tide_con_path), 'tide_cache.parquet')
            try:
                if os.path.exists(cache_path):
                    tide_df = pd.read_parquet(cache_path)
                    # Align with main data time index
                    main_close = self.raw_data_cache['close']  # [N, T]
                    T_main = main_close.shape[2] if main_close.dim() == 3 else main_close.shape[1]
                    pivot = tide_df.pivot(index='time', columns='address', values='tide_speed')
                    pivot = pivot.ffill().fillna(0.0)
                    t = torch.tensor(pivot.values.T, dtype=torch.float32, device=device)
                    # Pad or trim to match main tensor T
                    N_t, T_t = t.shape
                    N_main = main_close.shape[0]
                    if T_t > T_main:
                        t = t[:N_main, :T_main]
                    elif T_t < T_main:
                        pad = torch.zeros(N_main, T_main - T_t, device=device)
                        t = torch.cat([pad, t[:N_main]], dim=1)
                    else:
                        t = t[:N_main]
                    self.raw_data_cache['tide_speed'] = t
                    self._has_tide = True
                    print(f"Tide factor: loaded from cache ({len(tide_df)} rows)")
                else:
                    tide_con = duckdb.connect(self._tide_con_path, read_only=True)
                    self._compute_tide_factor(tide_con, device)
                    tide_con.close()
                    if 'tide_speed' in self.raw_data_cache:
                        self._has_tide = True
            except Exception as e:
                print(f"Warning: tide factor failed: {e}")

        # Load cached high-frequency factors (jump, etc.)
        import os as _os
        data_dir = _os.path.dirname(self.db_path)
        for cache_name, key in [('jump_cache.parquet', 'jump_ratio'),
                                 ('cvar_cache.parquet', 'cvar_5pct'),
                                 ('ambiguity_cache.parquet', 'amb_vol_corr'),
                                 ('ret_vol_cov_cache.parquet', 'ret_vol_cov'),
                                 ('follow_crowd_cache.parquet', 'follow_crowd'),
                                 ('lone_goose_cache.parquet', 'lone_goose'),
                                 ('dazzle_cache.parquet', 'dazzle_ret'),
                                 ('hurst_cache.parquet', 'hurst'),
                                 ('price_oi_corr_cache.parquet', 'price_oi_corr'),
                                 ('morning_fog_cache.parquet', 'morning_fog'),
                                 ('member_pos_cache.parquet', 'ls_raw'),
                                 ('warehouse_cache.parquet', 'warehouse'),
                                 ('inventory_cache.parquet', 'inventory'),
                                 ('long_conc_cache.parquet', 'long_conc_std'),
                                 ('shading_tree_cache.parquet', 'shading_tree'),
                                 ('tgd_cache.parquet', 'tgd'),
                                 ('inv_lunar_yoy_cache.parquet', 'inv_lunar_yoy'),
                                 ('net_support_vol_cache.parquet', 'net_support_vol'),
                                 ('ret_skew_cache.parquet', 'ret_skew'),
                                 ('short_weighted_cache.parquet', 'short_weighted')]:
            cache_file = _os.path.join(data_dir, cache_name)
            if _os.path.exists(cache_file):
                try:
                    cache_df = pd.read_parquet(cache_file)
                    main_close = self.raw_data_cache['close']
                    N_main, T_main = main_close.shape
                    # Reindex pivot to match main data's address/time ordering
                    # Get main addresses from the OHLCV pivot
                    main_df = df  # df from _query_main_contracts
                    main_times = sorted(main_df['time'].unique())
                    main_addrs = sorted(main_df['address'].unique())
                    pivot = cache_df.pivot(index='time', columns='address', values=key)
                    pivot = pivot.reindex(index=main_times, columns=main_addrs)
                    pivot = pivot.ffill().fillna(0.0)
                    t = torch.tensor(pivot.values.T, dtype=torch.float32, device=device)
                    # Should now be [N_main, T_main]
                    self.raw_data_cache[key] = t[:N_main, :T_main]
                    print(f"Cached factor '{key}': loaded ({len(cache_df)} rows)")
                except Exception as e:
                    print(f"Warning: failed to load {cache_name}: {e}")

        # Compute term-structure: first build raw tensors, then compute factors
        if self._has_term_structure:
            try:
                self._compute_term_structure(device)
            except Exception as e:
                print(f"Warning: term structure tensor build failed: {e}")
                self._has_term_structure = False

        # Compute all registered factors via registry
        try:
            from .factor_registry import get_registry
            from . import factors_lib  # noqa: F401 — triggers auto-discovery
            registry = get_registry()
            registry.auto_discover()
            reg_tensor, reg_names = registry.compute_group(self.raw_data_cache, '1d')
            if reg_tensor is not None:
                T_main = self.feat_tensor.shape[2]
                T_reg = reg_tensor.shape[2]
                N_main = self.feat_tensor.shape[0]
                N_reg = reg_tensor.shape[0]
                if T_reg >= T_main:
                    if N_reg >= N_main:
                        reg_aligned = reg_tensor[:N_main, :, :T_main]
                    elif N_reg < N_main:
                        # Pad missing products with zeros
                        pad = torch.zeros(N_main - N_reg, reg_tensor.shape[1], T_main,
                                          device=reg_tensor.device)
                        reg_aligned = torch.cat(
                            [reg_tensor[:, :, :T_main], pad], dim=0)
                    self.feat_tensor = torch.cat(
                        [self.feat_tensor, reg_aligned], dim=1)
                    print(f"Registry factors: +{len(reg_names)} dims "
                          f"({', '.join(reg_names)})")
                else:
                    print(f"Warning: registry factor shape mismatch "
                          f"(T: {T_reg} vs {T_main}), skipping")
        except Exception as e:
            print(f"Warning: registry factor computation failed: {e}")

        self.dates = sorted(df['time'].unique().tolist())

        # Target returns (next-bar return)
        close = self.raw_data_cache['close']
        next_close = torch.cat([close[:, 1:], torch.zeros_like(close[:, :1])], dim=1)
        self.target_ret = (next_close - close) / (close + 1e-9)
        self.target_ret[:, -1] = 0.0

        # Clamp per timeframe profile
        profile = TimeframeProfile(self.timeframe, ModelConfig.ASSET_CLASS)
        self.target_ret = torch.clamp(self.target_ret, -profile.ret_clamp, profile.ret_clamp)
        self.target_ret = torch.nan_to_num(self.target_ret, nan=0.0)

        # Build rebalance mask for intraday timeframes:
        # Only allow position changes at the first bar of each trading day.
        # This enables "1h features, daily rebalance" (Plan B).
        import numpy as np
        if profile.bars_per_day > 1:
            ts_index = pd.DatetimeIndex(self.dates)
            day_strs = np.array([str(d) for d in ts_index.date])
            is_new_day = np.concatenate([[True], day_strs[1:] != day_strs[:-1]])
            self.rebalance_mask = torch.tensor(
                is_new_day.astype(np.float32), device=device
            )  # [T]
            rebal_count = int(is_new_day.sum())
            print(f"Rebalance mask: {rebal_count} rebalance points "
                  f"out of {len(self.dates)} bars")
        else:
            self.rebalance_mask = torch.ones(
                len(self.dates), device=device
            )  # daily: rebalance every bar

        self.raw_data_cache['rebalance_mask'] = self.rebalance_mask

        # IC-screened feature selection: keep only the selected factors
        if not self.skip_ic_screening:
            from .features_v2 import FEATURES_V3_LIST, FEATURES_V3_FULL
            if self.feat_tensor.shape[1] == len(FEATURES_V3_FULL):
                full_names = FEATURES_V3_FULL
                selected_names = FEATURES_V3_LIST
                selected_indices = [full_names.index(n) for n in selected_names if n in full_names]
                if selected_indices:
                    self.feat_tensor = self.feat_tensor[:, selected_indices, :]
                    print(f"IC screening: {len(full_names)} → {len(selected_indices)} features")

        print(f"Data Ready. Shape: {self.feat_tensor.shape}")

        # Temporal train/test split
        time_steps = self.feat_tensor.shape[2]
        split_idx = int(time_steps * self.train_ratio)

        self.train_feat = self.feat_tensor[:, :, :split_idx]
        self.test_feat = self.feat_tensor[:, :, split_idx:]
        self.train_ret = self.target_ret[:, :split_idx]
        self.test_ret = self.target_ret[:, split_idx:]
        self.train_raw = {k: v[:, :split_idx] if v.dim() > 1 else v[:split_idx]
                          for k, v in self.raw_data_cache.items()}
        self.test_raw = {k: v[:, split_idx:] if v.dim() > 1 else v[split_idx:]
                         for k, v in self.raw_data_cache.items()}

        print(f"Split: train={split_idx} steps, test={time_steps - split_idx} steps")

    def _query_main_contracts(self, con, minutes, limit_tokens):
        """Query DuckDB for main-contract OHLCV, optionally aggregated."""

        # Product filter — applied in daily_oi CTE and in outer queries
        product_filter = ""
        daily_oi_filter = ""
        if self.products:
            plist = ", ".join(f"'{p}'" for p in self.products)
            product_filter = ""  # Filtering done in daily_oi CTE
            daily_oi_filter = f"WHERE product_id IN ({plist})"
        if self.end_date:
            date_clause = f"AND datetime <= '{self.end_date} 23:59:59'"
            if daily_oi_filter:
                daily_oi_filter += f" {date_clause}"
            else:
                daily_oi_filter = f"WHERE datetime <= '{self.end_date} 23:59:59'"

        # Main contract CTE (per product per day, pick highest avg close_oi)
        main_cte = f"""
            WITH daily_oi AS (
                SELECT product_id, exchange, symbol,
                       DATE_TRUNC('day', datetime) as day,
                       AVG(close_oi) as avg_oi
                FROM kline_1min
                {daily_oi_filter}
                GROUP BY product_id, exchange, symbol, day
            ),
            main_contracts AS (
                SELECT day, product_id, exchange, symbol,
                       ROW_NUMBER() OVER (
                           PARTITION BY product_id ORDER BY day, avg_oi DESC
                       ) as rn_dummy,
                       ROW_NUMBER() OVER (
                           PARTITION BY product_id, day ORDER BY avg_oi DESC
                       ) as rn
                FROM daily_oi
            )
        """

        if minutes >= 1440:
            # Daily aggregation with intraday microstructure columns
            sql = f"""
                {main_cte},
                vol_30 AS (
                    SELECT
                        mc2.product_id || '.' || mc2.exchange as address,
                        mc2.day as day,
                        TIME_BUCKET(INTERVAL '30 minutes', k2.datetime) as bucket,
                        SUM(k2.volume)::DOUBLE as vol_30
                    FROM main_contracts mc2
                    JOIN kline_1min k2 ON k2.symbol = mc2.symbol
                        AND DATE_TRUNC('day', k2.datetime) = mc2.day
                    WHERE mc2.rn = 1 {product_filter}
                    GROUP BY mc2.product_id || '.' || mc2.exchange, mc2.day, bucket
                ),
                max_vol_30 AS (
                    SELECT address, day, MAX(vol_30) as max_30min_vol
                    FROM vol_30
                    GROUP BY address, day
                ),
                vol_p90 AS (
                    SELECT mc4.product_id || '.' || mc4.exchange as address,
                           mc4.day,
                           QUANTILE_CONT(k4.volume, 0.9) as vol_p90
                    FROM main_contracts mc4
                    JOIN kline_1min k4 ON k4.symbol = mc4.symbol
                        AND DATE_TRUNC('day', k4.datetime) = mc4.day
                    WHERE mc4.rn = 1 {product_filter}
                    GROUP BY mc4.product_id || '.' || mc4.exchange, mc4.day
                ),
                flow_in_bars AS (
                    SELECT mc7.product_id || '.' || mc7.exchange as address,
                           mc7.day,
                           k7.volume::DOUBLE as vol,
                           k7.close::DOUBLE as cls,
                           LAG(k7.close) OVER (
                               PARTITION BY mc7.product_id, k7.symbol, mc7.day
                               ORDER BY k7.datetime) as prev_close
                    FROM main_contracts mc7
                    JOIN kline_1min k7 ON k7.symbol = mc7.symbol
                        AND DATE_TRUNC('day', k7.datetime) = mc7.day
                    WHERE mc7.rn = 1 {product_filter}
                ),
                flow_in AS (
                    SELECT address, day,
                           SUM(vol * cls * SIGN(cls - prev_close))
                               / NULLIF(SUM(vol * cls), 0) as flow_in_ratio
                    FROM flow_in_bars
                    WHERE prev_close IS NOT NULL
                    GROUP BY address, day
                ),
                inflect_bars AS (
                    SELECT mc6.product_id || '.' || mc6.exchange as address,
                           mc6.day,
                           k6.close - k6.open as bar_move,
                           LAG(k6.close - k6.open) OVER (
                               PARTITION BY mc6.product_id, k6.symbol, mc6.day
                               ORDER BY k6.datetime) as prev_move
                    FROM main_contracts mc6
                    JOIN kline_1min k6 ON k6.symbol = mc6.symbol
                        AND DATE_TRUNC('day', k6.datetime) = mc6.day
                    WHERE mc6.rn = 1 {product_filter}
                ),
                inflect AS (
                    SELECT address, day,
                           SUM(CASE WHEN bar_move * prev_move < 0 THEN 1 ELSE 0 END)::DOUBLE
                               / NULLIF(COUNT(*), 0) as inflection_ratio
                    FROM inflect_bars
                    WHERE prev_move IS NOT NULL
                    GROUP BY address, day
                ),
                intraday_rsi AS (
                    SELECT mc5.product_id || '.' || mc5.exchange as address,
                           mc5.day,
                           AVG(gain) / NULLIF(AVG(abs_chg), 0) * 100 as intraday_rsi
                    FROM main_contracts mc5
                    JOIN LATERAL (
                        SELECT GREATEST(k5.close - LAG(k5.close) OVER (ORDER BY k5.datetime), 0) as gain,
                               ABS(k5.close - LAG(k5.close) OVER (ORDER BY k5.datetime)) as abs_chg
                        FROM kline_1min k5
                        WHERE k5.symbol = mc5.symbol
                            AND DATE_TRUNC('day', k5.datetime) = mc5.day
                    ) rsi_sub ON TRUE
                    WHERE mc5.rn = 1 {product_filter}
                        AND abs_chg IS NOT NULL
                    GROUP BY mc5.product_id || '.' || mc5.exchange, mc5.day
                ),
                price_vol AS (
                    SELECT mc3.product_id || '.' || mc3.exchange as address,
                           mc3.day, k3.close as price_level,
                           SUM(k3.volume)::DOUBLE as level_vol
                    FROM main_contracts mc3
                    JOIN kline_1min k3 ON k3.symbol = mc3.symbol
                        AND DATE_TRUNC('day', k3.datetime) = mc3.day
                    WHERE mc3.rn = 1 {product_filter}
                    GROUP BY mc3.product_id || '.' || mc3.exchange, mc3.day, k3.close
                ),
                vsp AS (
                    SELECT address, day, price_level as vsp_price,
                           ROW_NUMBER() OVER (
                               PARTITION BY address, day ORDER BY level_vol DESC
                           ) as rn
                    FROM price_vol
                )
                SELECT
                    mc.product_id || '.' || mc.exchange as address,
                    mc.day as time,
                    FIRST(k.open ORDER BY k.datetime) as open,
                    MAX(k.high) as high,
                    MIN(k.low) as low,
                    LAST(k.close ORDER BY k.datetime) as close,
                    SUM(k.volume)::DOUBLE as volume,
                    (SUM(k.volume) * LAST(k.close ORDER BY k.datetime))::DOUBLE as liquidity,
                    0.0 as fdv,
                    -- Intraday microstructure columns
                    SUM(k.volume * k.close) / NULLIF(SUM(k.volume), 0) as vwap,
                    LAST(k.close_oi ORDER BY k.datetime) - FIRST(k.close_oi ORDER BY k.datetime) as oi_change,
                    LAST(k.close_oi ORDER BY k.datetime)::DOUBLE as total_oi,
                    SUM(CASE WHEN EXTRACT(HOUR FROM k.datetime) < 13 THEN k.volume ELSE 0 END)::DOUBLE as am_volume,
                    SUM(CASE WHEN EXTRACT(HOUR FROM k.datetime) >= 13 THEN k.volume ELSE 0 END)::DOUBLE as pm_volume,
                    mv.max_30min_vol,
                    SUM(CASE WHEN EXTRACT(HOUR FROM k.datetime) >= 14 THEN k.volume ELSE 0 END)::DOUBLE as last_hour_vol,
                    SUM(CASE WHEN EXTRACT(HOUR FROM k.datetime) < 10 THEN k.volume ELSE 0 END)::DOUBLE as first_hour_vol,
                    AVG(k.close)::DOUBLE as twap,
                    -- Volume Support Price position (b-type distribution)
                    (vp.vsp_price - MIN(k.low)) / NULLIF(MAX(k.high) - MIN(k.low), 0) as vsp_position,
                    -- Buy willingness: fraction of 1min bars where close > open
                    SUM(CASE WHEN k.close > k.open THEN 1 ELSE 0 END)::DOUBLE
                        / NULLIF(COUNT(*), 0) as buy_ratio,
                    -- Large order impact: avg return during high-volume minutes
                    AVG(CASE WHEN k.volume >= vp90.vol_p90
                        THEN LN(k.close / NULLIF(k.open, 0)) END) as large_order_ret,
                    -- Intraday RSI from 1min data (via rsi CTE)
                    rsi_cte.intraday_rsi,
                    -- Volume ratio: up-volume / down-volume
                    SUM(CASE WHEN k.close > k.open THEN k.volume ELSE 0 END)::DOUBLE
                        / NULLIF(SUM(CASE WHEN k.close <= k.open THEN k.volume ELSE 0 END), 0) as vol_ratio,
                    -- Inflection point ratio (via inflect CTE)
                    inflect_cte.inflection_ratio,
                    -- Flow in ratio from 1min data (via flow_in CTE)
                    flow_in_cte.flow_in_ratio
                FROM main_contracts mc
                JOIN kline_1min k ON k.symbol = mc.symbol
                    AND DATE_TRUNC('day', k.datetime) = mc.day
                LEFT JOIN max_vol_30 mv ON mv.address = mc.product_id || '.' || mc.exchange
                    AND mv.day = mc.day
                LEFT JOIN vsp vp ON vp.address = mc.product_id || '.' || mc.exchange
                    AND vp.day = mc.day AND vp.rn = 1
                LEFT JOIN vol_p90 vp90 ON vp90.address = mc.product_id || '.' || mc.exchange
                    AND vp90.day = mc.day
                LEFT JOIN intraday_rsi rsi_cte ON rsi_cte.address = mc.product_id || '.' || mc.exchange
                    AND rsi_cte.day = mc.day
                LEFT JOIN inflect inflect_cte ON inflect_cte.address = mc.product_id || '.' || mc.exchange
                    AND inflect_cte.day = mc.day
                LEFT JOIN flow_in flow_in_cte ON flow_in_cte.address = mc.product_id || '.' || mc.exchange
                    AND flow_in_cte.day = mc.day
                WHERE mc.rn = 1 {product_filter}
                GROUP BY mc.product_id || '.' || mc.exchange, mc.day, mv.max_30min_vol, vp.vsp_price, vp90.vol_p90, rsi_cte.intraday_rsi, inflect_cte.inflection_ratio, flow_in_cte.flow_in_ratio
                ORDER BY address, time
            """
        elif minutes == 1:
            # Raw 1min — no aggregation needed
            sql = f"""
                {main_cte}
                SELECT
                    mc.product_id || '.' || mc.exchange as address,
                    k.datetime as time,
                    k.open, k.high, k.low, k.close,
                    k.volume::DOUBLE as volume,
                    (k.volume * k.close)::DOUBLE as liquidity,
                    0.0 as fdv
                FROM main_contracts mc
                JOIN kline_1min k ON k.symbol = mc.symbol
                    AND DATE_TRUNC('day', k.datetime) = mc.day
                WHERE mc.rn = 1 {product_filter}
                ORDER BY address, time
            """
        else:
            # Intraday aggregation (5m, 15m, 30m, 1h)
            sql = f"""
                {main_cte}
                SELECT
                    mc.product_id || '.' || mc.exchange as address,
                    TIME_BUCKET(INTERVAL '{minutes} minutes', k.datetime) as time,
                    FIRST(k.open ORDER BY k.datetime) as open,
                    MAX(k.high) as high,
                    MIN(k.low) as low,
                    LAST(k.close ORDER BY k.datetime) as close,
                    SUM(k.volume)::DOUBLE as volume,
                    (SUM(k.volume) * LAST(k.close ORDER BY k.datetime))::DOUBLE as liquidity,
                    0.0 as fdv
                FROM main_contracts mc
                JOIN kline_1min k ON k.symbol = mc.symbol
                    AND DATE_TRUNC('day', k.datetime) = mc.day
                WHERE mc.rn = 1 {product_filter}
                GROUP BY address, TIME_BUCKET(INTERVAL '{minutes} minutes', k.datetime)
                ORDER BY address, time
            """

        df = con.execute(sql).fetchdf()

        # Limit number of products by data volume
        if df['address'].nunique() > limit_tokens:
            top = df.groupby('address').size().nlargest(limit_tokens).index
            df = df[df['address'].isin(top)]

        return df

    def _query_secondary_contracts(self, con):
        """Query the top-2 contracts (by OI) daily close for term structure.

        Two-pass approach to avoid expensive 4-table JOIN:
        1. Get daily close for ALL contracts (light aggregation)
        2. Rank by OI and pivot in Python

        Returns DataFrame with columns:
            address, time, F1_close, F2_close, F1_symbol, F2_symbol
        """
        product_filter = ""
        if self.products:
            plist = ", ".join(f"'{p}'" for p in self.products)
            product_filter = f"WHERE product_id IN ({plist})"

        sql = f"""
            SELECT
                product_id || '.' || exchange as address,
                symbol,
                DATE_TRUNC('day', datetime) as time,
                AVG(close_oi) as avg_oi,
                LAST(close ORDER BY datetime) as daily_close
            FROM kline_1min
            {product_filter}
            GROUP BY product_id || '.' || exchange, symbol,
                     DATE_TRUNC('day', datetime)
            ORDER BY address, time, avg_oi DESC
        """
        df = con.execute(sql).fetchdf()
        if df.empty:
            return pd.DataFrame()

        # Rank by OI within (address, time), take top 2
        df['rn'] = df.groupby(['address', 'time'])['avg_oi'].rank(
            ascending=False, method='first').astype(int)
        top2 = df[df['rn'] <= 2].copy()

        # Pivot: separate F1 (rn=1) and F2 (rn=2)
        f1 = top2[top2['rn'] == 1][['address', 'time', 'daily_close', 'symbol']].rename(
            columns={'daily_close': 'F1_close', 'symbol': 'F1_symbol'})
        f2 = top2[top2['rn'] == 2][['address', 'time', 'daily_close', 'symbol']].rename(
            columns={'daily_close': 'F2_close', 'symbol': 'F2_symbol'})

        merged = f1.merge(f2, on=['address', 'time'], how='inner')
        return merged

    def _compute_tide_factor(self, con, device):
        """Compute volume tide speed per product per day from 1-min data.

        For each trading day:
        1. Domain volume = 9-bar rolling sum (±4 bars)
        2. Peak = max domain volume bar
        3. Tide-in (m) = min domain volume before peak
        4. Tide-out (n) = min domain volume after peak
        5. Tide speed = (C_n - C_m) / C_m / (n - m)

        Runs per-product queries to avoid full-table JOIN explosion.
        """
        import pandas as pd

        product_filter_sql = ""
        if self.products:
            plist = ", ".join(f"'{p}'" for p in self.products)
            product_filter_sql = f"AND product_id IN ({plist})"

        products = con.execute(f"""
            SELECT DISTINCT product_id, exchange
            FROM kline_1min WHERE 1=1 {product_filter_sql}
        """).fetchdf()

        results = []
        for _, row in products.iterrows():
            pid, exch = row['product_id'], row['exchange']
            addr = f"{pid}.{exch}"
            try:
                df = con.execute(f"""
                    WITH ranked_contracts AS (
                        SELECT symbol, AVG(close_oi) as avg_oi
                        FROM kline_1min
                        WHERE product_id = '{pid}' AND exchange = '{exch}'
                        GROUP BY symbol
                        ORDER BY avg_oi DESC LIMIT 1
                    ),
                    local_vol AS (
                        SELECT DATE_TRUNC('day', datetime) as day,
                               ROW_NUMBER() OVER (
                                   PARTITION BY DATE_TRUNC('day', datetime)
                                   ORDER BY datetime) as bar_idx,
                               close,
                               SUM(volume) OVER (
                                   PARTITION BY DATE_TRUNC('day', datetime)
                                   ORDER BY datetime
                                   ROWS BETWEEN 4 PRECEDING AND 4 FOLLOWING
                               ) as domain_vol
                        FROM kline_1min
                        WHERE product_id = '{pid}' AND exchange = '{exch}'
                            AND symbol = (SELECT symbol FROM ranked_contracts)
                    ),
                    peak AS (
                        SELECT day, ARG_MAX(bar_idx, domain_vol) as peak_idx
                        FROM local_vol GROUP BY day
                    ),
                    tm AS (
                        SELECT lv.day, ARG_MIN(lv.bar_idx, lv.domain_vol) as m_idx
                        FROM local_vol lv JOIN peak p ON p.day = lv.day
                        WHERE lv.bar_idx >= 5 AND lv.bar_idx < p.peak_idx
                        GROUP BY lv.day
                    ),
                    tn AS (
                        SELECT lv.day, ARG_MIN(lv.bar_idx, lv.domain_vol) as n_idx
                        FROM local_vol lv JOIN peak p ON p.day = lv.day
                        WHERE lv.bar_idx > p.peak_idx
                        GROUP BY lv.day
                    )
                    SELECT tm.day as time,
                           (cn.close - cm.close) / NULLIF(cm.close, 0)
                               / NULLIF(tn.n_idx - tm.m_idx, 0) as tide_speed
                    FROM tm
                    JOIN tn ON tn.day = tm.day
                    JOIN local_vol cm ON cm.day = tm.day AND cm.bar_idx = tm.m_idx
                    JOIN local_vol cn ON cn.day = tn.day AND cn.bar_idx = tn.n_idx
                    ORDER BY time
                """).fetchdf()
                if not df.empty:
                    df['address'] = addr
                    results.append(df)
            except Exception:
                continue

        if not results:
            return

        all_df = pd.concat(results, ignore_index=True)

        def pivot_to_tensor(col):
            pivot = all_df.pivot(index='time', columns='address', values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=device)

        self.raw_data_cache['tide_speed'] = pivot_to_tensor('tide_speed')
        print(f"Tide factor: {len(all_df)} rows, "
              f"{all_df['address'].nunique()} products")

    def _compute_term_structure(self, device):
        """Compute F1/F2 close, days-to-expiry, and contract-gap tensors.

        Uses self._secondary_df (pre-loaded in load_data).
        Adds to self.raw_data_cache:
            F1_close, F2_close, days_to_expiry, contract_gap_days
        """
        df2 = self._secondary_df
        if df2 is None or df2.empty:
            print("Warning: no secondary contract data, skipping term structure")
            return

        # Parse expiry month from symbol, e.g. SHFE.cu2604 → 2026-04
        def _expiry_date(sym):
            m = re.search(r'(\d{4})$', str(sym))
            if not m:
                return None
            code = m.group(1)  # e.g. '2604'
            yr = 2000 + int(code[:2])
            mo = int(code[2:])
            # Approximate expiry as 15th of the month
            return pd.Timestamp(year=yr, month=mo, day=15)

        df2['F1_expiry'] = df2['F1_symbol'].apply(_expiry_date)
        df2['F2_expiry'] = df2['F2_symbol'].apply(_expiry_date)
        df2['time'] = pd.to_datetime(df2['time'])

        # Days to near-month expiry (n0)
        df2['days_to_expiry'] = (df2['F1_expiry'] - df2['time']).dt.days.clip(lower=1)
        # Days between F1 and F2 expiry (n1)
        df2['contract_gap_days'] = (df2['F2_expiry'] - df2['F1_expiry']).dt.days.clip(lower=1)

        # Pivot to tensors aligned with main data
        def pivot_to_tensor(col):
            pivot = df2.pivot(index='time', columns='address', values=col)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=device)

        # Align columns to match main data's address ordering
        addresses_main = list(
            pd.DataFrame({'address': df2['address'].unique()}).sort_values('address')['address']
        )

        self.raw_data_cache['F1_close'] = pivot_to_tensor('F1_close')
        self.raw_data_cache['F2_close'] = pivot_to_tensor('F2_close')
        self.raw_data_cache['days_to_expiry'] = pivot_to_tensor('days_to_expiry')
        self.raw_data_cache['contract_gap_days'] = pivot_to_tensor('contract_gap_days')

        print(f"Term structure: F1/F2 loaded, "
              f"{(df2['F2_close'] > 0).sum()}/{len(df2)} rows with valid F2")
