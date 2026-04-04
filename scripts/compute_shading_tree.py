"""
Compute "午蔽古木" (Shading Tree) factor from 1-minute kline data.

Algorithm (per product per day):
    1. ret_t = close_t / close_{t-1} - 1  (1-min return)
       voldiff_t = volume_t - volume_{t-1}  (1-min volume diff)
    2. OLS on minutes 6..240:
       ret_t = α₀ + α₁·voldiff_t + α₂·voldiff_{t-1} + ... + α₆·voldiff_{t-5} + ε_t
    3. Record t-intercept and F-all from the regression
    4. abst_intercept = |t-intercept|
       If F-all < cross-sectional mean(F-all), flip sign: abst_intercept *= -1

Reference:
    曹有梅, 2023. 驱动个股价格变化的因素分解与"花隐仙间"因子. 方正证券.

Usage:
    python scripts/compute_shading_tree.py
    python scripts/compute_shading_tree.py --products rb al
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH",
                         os.path.expanduser("~/quant/tick_data/kline_1min.duckdb"))


def compute_shading_tree(products=None):
    import duckdb

    con = duckdb.connect(DUCKDB_PATH, read_only=True)

    # Get main contract mapping
    product_filter = ""
    if products:
        plist = ", ".join(f"'{p}'" for p in products)
        product_filter = f"WHERE product_id IN ({plist})"

    sql = f"""
        WITH daily_oi AS (
            SELECT product_id, exchange, symbol,
                   DATE_TRUNC('day', datetime) as day,
                   AVG(close_oi) as avg_oi
            FROM kline_1min
            {product_filter}
            GROUP BY product_id, exchange, symbol, day
        ),
        main_contracts AS (
            SELECT day, product_id, exchange, symbol,
                   ROW_NUMBER() OVER (
                       PARTITION BY product_id, day ORDER BY avg_oi DESC
                   ) as rn
            FROM daily_oi
        )
        SELECT
            mc.product_id || '.' || mc.exchange as address,
            mc.day,
            k.datetime,
            k.close,
            k.volume
        FROM main_contracts mc
        JOIN kline_1min k ON k.symbol = mc.symbol
            AND DATE_TRUNC('day', k.datetime) = mc.day
        WHERE mc.rn = 1
        ORDER BY address, mc.day, k.datetime
    """

    print("Querying 1-min data from DuckDB...")
    df = con.execute(sql).fetchdf()
    con.close()
    print(f"Loaded {len(df)} rows, {df['address'].nunique()} products, "
          f"{df['day'].nunique()} days")

    results = []
    n_lags = 6  # voldiff_t, voldiff_{t-1}, ..., voldiff_{t-5}

    for (address, day), group in df.groupby(['address', 'day']):
        group = group.sort_values('datetime').reset_index(drop=True)
        if len(group) < 20:
            continue

        close = group['close'].values.astype(np.float64)
        volume = group['volume'].values.astype(np.float64)

        # Step 1: compute ret and voldiff
        ret = close[1:] / close[:-1] - 1.0
        voldiff = np.diff(volume)

        # Need at least n_lags + 1 valid bars after differencing
        # Use bars 5 onwards (0-indexed), which corresponds to minute 6+ in paper
        # (first bar is index 0, after diff we start from index n_lags)
        start_idx = n_lags  # skip first n_lags for lag construction
        if start_idx >= len(ret):
            continue

        y = ret[start_idx:]
        n_obs = len(y)
        if n_obs < n_lags + 5:  # need enough observations for regression
            continue

        # Step 2: build design matrix [intercept, voldiff_t, voldiff_{t-1}, ..., voldiff_{t-5}]
        X = np.ones((n_obs, 1 + n_lags), dtype=np.float64)
        for lag in range(n_lags):
            X[:, 1 + lag] = voldiff[start_idx - lag: start_idx - lag + n_obs]

        # OLS: beta = (X'X)^{-1} X'y
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            continue

        residuals = y - X @ beta
        dof = n_obs - (1 + n_lags)
        if dof <= 0:
            continue

        # Step 3: t-statistic for intercept, F-statistic for full model
        mse = np.sum(residuals ** 2) / dof
        try:
            cov_beta = mse * np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            continue

        se_intercept = np.sqrt(max(cov_beta[0, 0], 1e-30))
        t_intercept = beta[0] / se_intercept

        # F-statistic: (SSR / k) / (SSE / (n - k - 1))
        y_mean = y.mean()
        ss_total = np.sum((y - y_mean) ** 2)
        ss_resid = np.sum(residuals ** 2)
        ss_reg = ss_total - ss_resid
        k = n_lags  # number of regressors (excluding intercept)
        f_all = (ss_reg / max(k, 1)) / (ss_resid / max(dof, 1)) if ss_resid > 1e-30 else 0.0

        results.append({
            'time': pd.Timestamp(day),
            'address': address,
            't_intercept': float(t_intercept),
            'f_all': float(f_all),
        })

    if not results:
        print("No results computed.")
        return

    result_df = pd.DataFrame(results)
    print(f"Computed {len(result_df)} product-day observations")

    # Step 4: cross-sectional adjustment
    # abst_intercept = |t_intercept|
    # If F_all < cross-sectional mean(F_all) for that day, flip sign to -1
    result_df['abst_intercept'] = result_df['t_intercept'].abs()

    daily_f_mean = result_df.groupby('time')['f_all'].transform('mean')
    mask_low_f = result_df['f_all'] < daily_f_mean
    result_df.loc[mask_low_f, 'abst_intercept'] *= -1

    # The final factor value
    result_df['shading_tree'] = result_df['abst_intercept']

    # Save cache
    out_df = result_df[['time', 'address', 'shading_tree']].copy()
    out_df = out_df.sort_values(['address', 'time']).reset_index(drop=True)

    data_dir = os.path.dirname(DUCKDB_PATH)
    cache_path = os.path.join(data_dir, "shading_tree_cache.parquet")
    out_df.to_parquet(cache_path, index=False)

    print(f"Saved: {cache_path}")
    print(f"  {len(out_df)} rows, {out_df['address'].nunique()} products, "
          f"{out_df['time'].nunique()} days")
    print(f"  Range: {out_df['time'].min().date()} → {out_df['time'].max().date()}")
    print(f"  Value stats: mean={out_df['shading_tree'].mean():.4f}, "
          f"std={out_df['shading_tree'].std():.4f}")
    return cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute shading tree factor")
    parser.add_argument("--products", nargs='+', default=None,
                        help="Product IDs to compute (default: all)")
    args = parser.parse_args()
    compute_shading_tree(products=args.products)
