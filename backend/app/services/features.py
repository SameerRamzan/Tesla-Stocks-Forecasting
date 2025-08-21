"""Feature engineering service for Tesla stock data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert
import holidays

from backend.app.models.database import PriceData, FeatureData
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Service for engineering features from raw price data."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.symbol = settings.YFINANCE_SYMBOL
        self.us_holidays = holidays.UnitedStates()

    async def compute_and_store_features(
        self,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 100,  # Need more history for feature computation
    ) -> Dict[str, Any]:
        """
        Compute features from price data and store in database.
        
        Args:
            interval: Data interval
            start_date: Start date for feature computation
            end_date: End date for feature computation
            lookback_days: Days of history needed for feature computation
            
        Returns:
            Dictionary with computation results
        """
        try:
            logger.info(f"Computing features for {self.symbol} interval {interval}")

            # Get price data
            price_data = await self._get_price_data(interval, start_date, end_date, lookback_days)
            
            if price_data.empty:
                logger.warning("No price data available for feature computation")
                return {
                    "status": "warning",
                    "message": "No price data available",
                    "features_computed": 0
                }

            # Compute features
            features_df = self._compute_features(price_data)
            
            # Store features in database
            result = await self._upsert_features(features_df, interval)
            
            logger.info(f"Successfully computed {len(features_df)} feature records")
            
            return {
                "status": "success",
                "symbol": self.symbol,
                "interval": interval,
                "features_computed": len(features_df),
                **result
            }

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "features_computed": 0
            }

    async def _get_price_data(
        self,
        interval: str,
        start_date: Optional[str],
        end_date: Optional[str],
        lookback_days: int
    ) -> pd.DataFrame:
        """Get price data from database."""
        
        # Calculate extended date range for feature computation
        if start_date:
            # Need extra history for rolling features
            extended_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        else:
            extended_start = (datetime.now() - timedelta(days=lookback_days + 60)).strftime("%Y-%m-%d")
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        stmt = select(PriceData).where(
            and_(
                PriceData.symbol == self.symbol,
                PriceData.interval == interval,
                PriceData.ts >= extended_start,
                PriceData.ts <= end_date
            )
        ).order_by(PriceData.ts)

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        if not records:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([{
            "ts": r.ts,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "adj_close": r.adj_close,
            "volume": r.volume,
            "dividends": r.dividends,
            "stock_splits": r.stock_splits
        } for r in records])

        df.set_index("ts", inplace=True)
        df.sort_index(inplace=True)
        
        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from price data."""
        
        features_df = df.copy()
        
        # Returns
        features_df = self._compute_returns(features_df)
        
        # Rolling statistics
        features_df = self._compute_rolling_stats(features_df)
        
        # Technical indicators
        features_df = self._compute_technical_indicators(features_df)
        
        # Calendar features
        features_df = self._compute_calendar_features(features_df)
        
        # Target variables (shifted future values)
        features_df = self._compute_targets(features_df)
        
        # Clean up: remove rows with insufficient history
        features_df = features_df.dropna()
        
        return features_df

    def _compute_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute return features."""
        
        # Simple returns
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_10d"] = df["close"].pct_change(10)
        df["return_20d"] = df["close"].pct_change(20)
        
        # Log returns
        df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
        
        return df

    def _compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling window statistics."""
        
        # Rolling means
        df["rolling_mean_5"] = df["close"].rolling(window=5).mean()
        df["rolling_mean_20"] = df["close"].rolling(window=20).mean()
        
        # Rolling standard deviations
        df["rolling_std_5"] = df["close"].rolling(window=5).std()
        df["rolling_std_20"] = df["close"].rolling(window=20).std()
        
        return df

    def _compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        
        # RSI (Relative Strength Index)
        df["rsi_14"] = self._compute_rsi(df["close"], window=14)
        
        # MACD
        macd_data = self._compute_macd(df["close"])
        df["macd"] = macd_data["macd"]
        df["macd_signal"] = macd_data["signal"]
        df["macd_histogram"] = macd_data["histogram"]
        
        # Bollinger Bands
        bb_data = self._compute_bollinger_bands(df["close"])
        df["bb_upper"] = bb_data["upper"]
        df["bb_lower"] = bb_data["lower"]
        df["bb_middle"] = bb_data["middle"]
        
        # ATR (Average True Range)
        df["atr_14"] = self._compute_atr(df, window=14)
        
        # OBV (On Balance Volume)
        df["obv"] = self._compute_obv(df["close"], df["volume"])
        
        return df

    def _compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Compute MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            "macd": macd,
            "signal": signal_line,
            "histogram": histogram
        }

    def _compute_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Compute Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        return {
            "middle": rolling_mean,
            "upper": rolling_mean + (rolling_std * num_std),
            "lower": rolling_mean - (rolling_std * num_std)
        }

    def _compute_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def _compute_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On Balance Volume."""
        price_change = prices.diff()
        obv = volume.copy()
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        return obv.cumsum()

    def _compute_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute calendar-based features."""
        
        # Extract datetime components
        df["day_of_week"] = df.index.dayofweek  # 0=Monday, 6=Sunday
        df["month"] = df.index.month
        
        # End of period indicators
        df["is_month_end"] = df.index.is_month_end
        df["is_quarter_end"] = df.index.is_quarter_end
        df["is_year_end"] = df.index.is_year_end
        
        # US holidays
        df["is_holiday"] = df.index.date.to_series().apply(lambda x: x in self.us_holidays).values
        
        return df

    def _compute_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute target variables (future values, avoiding leakage)."""
        
        # Future close prices
        df["target_close_1d"] = df["close"].shift(-1)
        df["target_close_5d"] = df["close"].shift(-5)
        
        # Future returns
        df["target_return_1d"] = df["return_1d"].shift(-1)
        df["target_return_5d"] = df["return_5d"].shift(-5)
        
        return df

    async def _upsert_features(self, features_df: pd.DataFrame, interval: str) -> Dict[str, int]:
        """Upsert feature data into database."""
        
        if features_df.empty:
            return {"features_inserted": 0, "features_updated": 0}

        # Prepare records for database
        records = []
        for ts, row in features_df.iterrows():
            record = {
                "symbol": self.symbol,
                "interval": interval,
                "ts": ts,
                "return_1d": self._safe_float(row.get("return_1d")),
                "return_5d": self._safe_float(row.get("return_5d")),
                "return_10d": self._safe_float(row.get("return_10d")),
                "return_20d": self._safe_float(row.get("return_20d")),
                "log_return_1d": self._safe_float(row.get("log_return_1d")),
                "rolling_mean_5": self._safe_float(row.get("rolling_mean_5")),
                "rolling_mean_20": self._safe_float(row.get("rolling_mean_20")),
                "rolling_std_5": self._safe_float(row.get("rolling_std_5")),
                "rolling_std_20": self._safe_float(row.get("rolling_std_20")),
                "rsi_14": self._safe_float(row.get("rsi_14")),
                "macd": self._safe_float(row.get("macd")),
                "macd_signal": self._safe_float(row.get("macd_signal")),
                "macd_histogram": self._safe_float(row.get("macd_histogram")),
                "bb_upper": self._safe_float(row.get("bb_upper")),
                "bb_lower": self._safe_float(row.get("bb_lower")),
                "bb_middle": self._safe_float(row.get("bb_middle")),
                "atr_14": self._safe_float(row.get("atr_14")),
                "obv": self._safe_float(row.get("obv")),
                "day_of_week": int(row.get("day_of_week", 0)),
                "month": int(row.get("month", 1)),
                "is_month_end": bool(row.get("is_month_end", False)),
                "is_quarter_end": bool(row.get("is_quarter_end", False)),
                "is_year_end": bool(row.get("is_year_end", False)),
                "is_holiday": bool(row.get("is_holiday", False)),
                "target_close_1d": self._safe_float(row.get("target_close_1d")),
                "target_close_5d": self._safe_float(row.get("target_close_5d")),
                "target_return_1d": self._safe_float(row.get("target_return_1d")),
                "target_return_5d": self._safe_float(row.get("target_return_5d")),
                "created_at": datetime.utcnow()
            }
            records.append(record)

        try:
            # Use PostgreSQL's INSERT ... ON CONFLICT for upsert
            stmt = insert(FeatureData).values(records)
            
            # Define what to do on conflict
            update_columns = {
                col.name: stmt.excluded[col.name] 
                for col in FeatureData.__table__.columns 
                if col.name not in ['symbol', 'interval', 'ts']
            }
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'interval', 'ts'],
                set_=update_columns
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return {
                "features_inserted": len(records),
                "features_updated": 0
            }
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error upserting features: {str(e)}")
            raise

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float, handling NaN and None."""
        if pd.isna(value) or value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None