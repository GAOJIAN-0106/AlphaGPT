import asyncpg
from loguru import logger
from .config import Config

class DBManager:
    def __init__(self):
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=Config.DB_DSN)
            logger.info("Database connection established.")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def init_schema(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    address TEXT PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    decimals INT,
                    chain TEXT,
                    last_updated TIMESTAMP DEFAULT NOW()
                );
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMP NOT NULL,
                    address TEXT NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    liquidity DOUBLE PRECISION,
                    fdv DOUBLE PRECISION,
                    source TEXT,
                    timeframe TEXT NOT NULL DEFAULT '1d',
                    PRIMARY KEY (time, address, timeframe)
                );
            """)

            try:
                await conn.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);")
                logger.info("Converted ohlcv to Hypertable.")
            except Exception:
                logger.warning("TimescaleDB extension not found, using standard Postgres.")

            await conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_address ON ohlcv (address);")

        # Migrate existing tables that lack the timeframe column
        await self._migrate_add_timeframe()

    async def _migrate_add_timeframe(self):
        """Add timeframe column to ohlcv if it doesn't exist (migration for existing DBs)."""
        async with self.pool.acquire() as conn:
            has_col = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'ohlcv' AND column_name = 'timeframe'
                );
            """)
            if has_col:
                return

            logger.info("Migrating ohlcv: adding timeframe column...")
            await conn.execute("ALTER TABLE ohlcv ADD COLUMN timeframe TEXT NOT NULL DEFAULT '1d';")

            # Recreate PK to include timeframe
            await conn.execute("ALTER TABLE ohlcv DROP CONSTRAINT IF EXISTS ohlcv_pkey;")
            await conn.execute("ALTER TABLE ohlcv ADD PRIMARY KEY (time, address, timeframe);")
            logger.info("Migration complete: ohlcv now includes timeframe in PK.")

    async def upsert_tokens(self, tokens):
        if not tokens: return
        async with self.pool.acquire() as conn:
            # tokens: list of (address, symbol, name, decimals, chain)
            await conn.executemany("""
                INSERT INTO tokens (address, symbol, name, decimals, chain, last_updated)
                VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (address) DO UPDATE 
                SET symbol = EXCLUDED.symbol, last_updated = NOW();
            """, tokens)

    async def batch_insert_ohlcv(self, records):
        if not records: return
        async with self.pool.acquire() as conn:
            try:
                await conn.copy_records_to_table(
                    'ohlcv',
                    records=records,
                    columns=['time', 'address', 'open', 'high', 'low', 'close',
                             'volume', 'liquidity', 'fdv', 'source', 'timeframe'],
                    timeout=60
                )
            except asyncpg.UniqueViolationError:
                pass # 忽略重复
            except Exception as e:
                logger.error(f"Batch insert error: {e}")