"""Data ingestion service for Tesla stock data using yfinance."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import yfinance as yf
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert

from backend.app.models.database import PriceData
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class DataIngestionService:
    """Service for ingesting Tesla stock data from yfinance."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.symbol = settings.YFINANCE_SYMBOL

    async def fetch_and_store_data(
        self,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Fetch Tesla data from yfinance and store in database.
        
        Args:
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            lookback_days: Number of days to look back if no start_date provided
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            # Calculate date range if not provided
            if not start_date:
                start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")

            logger.info(f"Fetching {self.symbol} data for interval {interval} from {start_date} to {end_date}")

            # Fetch data from yfinance
            ticker = yf.Ticker(self.symbol)
            
            # Get historical data
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,  # Get raw OHLC without adjustments
                prepost=False,      # Exclude pre/post market data
                threads=True
            )

            if hist_data.empty:
                logger.warning(f"No data returned for {self.symbol} in specified date range")
                return {
                    "status": "warning",
                    "message": "No data returned from yfinance",
                    "records_processed": 0,
                    "records_inserted": 0,
                    "records_updated": 0
                }

            # Get dividend and split data
            dividends = ticker.dividends
            splits = ticker.splits

            # Process the data
            processed_data = self._process_yfinance_data(hist_data, dividends, splits, interval)
            
            # Store in database
            result = await self._upsert_price_data(processed_data)
            
            logger.info(f"Successfully processed {len(processed_data)} records for {self.symbol}")
            
            return {
                "status": "success",
                "symbol": self.symbol,
                "interval": interval,
                "start_date": start_date,
                "end_date": end_date,
                "records_processed": len(processed_data),
                **result
            }

        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "records_processed": 0,
                "records_inserted": 0,
                "records_updated": 0
            }

    def _process_yfinance_data(
        self,
        hist_data: pd.DataFrame,
        dividends: pd.Series,
        splits: pd.Series,
        interval: str
    ) -> List[Dict[str, Any]]:
        """Process raw yfinance data into database-ready format."""
        
        processed_records = []
        
        for timestamp, row in hist_data.iterrows():
            # Convert timestamp to timezone-aware datetime
            if timestamp.tz is None:
                timestamp = timestamp.tz_localize('UTC')
            else:
                timestamp = timestamp.tz_convert('UTC')
            
            # Get dividend and split data for this timestamp
            dividend_amount = 0.0
            if not dividends.empty and timestamp.date() in dividends.index.date:
                dividend_amount = float(dividends[dividends.index.date == timestamp.date()].iloc[0])
            
            split_ratio = 1.0
            if not splits.empty and timestamp.date() in splits.index.date:
                split_ratio = float(splits[splits.index.date == timestamp.date()].iloc[0])

            record = {
                "symbol": self.symbol,
                "interval": interval,
                "ts": timestamp,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "adj_close": float(row["Adj Close"]),
                "volume": float(row["Volume"]),
                "dividends": dividend_amount,
                "stock_splits": split_ratio,
                "source_ingested_at": datetime.utcnow()
            }
            
            processed_records.append(record)
        
        return processed_records

    async def _upsert_price_data(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upsert price data records using PostgreSQL's ON CONFLICT."""
        
        if not records:
            return {"records_inserted": 0, "records_updated": 0}
        
        try:
            # Use PostgreSQL's INSERT ... ON CONFLICT for upsert
            stmt = insert(PriceData).values(records)
            
            # Define what to do on conflict (update all non-key columns)
            update_columns = {
                col.name: stmt.excluded[col.name] 
                for col in PriceData.__table__.columns 
                if col.name not in ['symbol', 'interval', 'ts']
            }
            
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'interval', 'ts'],
                set_=update_columns
            )
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            # For simplicity, we'll report all as inserted
            # In a real system, you might want to track actual inserts vs updates
            return {
                "records_inserted": len(records),
                "records_updated": 0
            }
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error upserting price data: {str(e)}")
            raise

    async def get_latest_data_timestamp(self, interval: str = "1d") -> Optional[datetime]:
        """Get the timestamp of the most recent data for the given interval."""
        
        try:
            stmt = select(PriceData.ts).where(
                and_(
                    PriceData.symbol == self.symbol,
                    PriceData.interval == interval
                )
            ).order_by(PriceData.ts.desc()).limit(1)
            
            result = await self.session.execute(stmt)
            latest_ts = result.scalar()
            
            return latest_ts
            
        except Exception as e:
            logger.error(f"Error getting latest timestamp: {str(e)}")
            return None

    async def get_data_quality_summary(self, interval: str = "1d") -> Dict[str, Any]:
        """Get data quality summary for the specified interval."""
        
        try:
            # Get basic statistics
            stmt = select(PriceData).where(
                and_(
                    PriceData.symbol == self.symbol,
                    PriceData.interval == interval
                )
            )
            
            result = await self.session.execute(stmt)
            records = result.scalars().all()
            
            if not records:
                return {"status": "no_data", "record_count": 0}
            
            # Convert to pandas for easy analysis
            df = pd.DataFrame([{
                "ts": r.ts,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume
            } for r in records])
            
            # Basic validation checks
            quality_checks = {
                "record_count": len(df),
                "date_range": {
                    "start": df["ts"].min().isoformat(),
                    "end": df["ts"].max().isoformat()
                },
                "missing_data": df.isnull().sum().to_dict(),
                "price_consistency": {
                    "low_high_violations": (df["low"] > df["high"]).sum(),
                    "low_close_violations": (df["low"] > df["close"]).sum(),
                    "low_open_violations": (df["low"] > df["open"]).sum(),
                    "high_close_violations": (df["high"] < df["close"]).sum(),
                    "high_open_violations": (df["high"] < df["open"]).sum(),
                },
                "negative_volume": (df["volume"] < 0).sum(),
                "zero_volume": (df["volume"] == 0).sum(),
            }
            
            return {
                "status": "success",
                "symbol": self.symbol,
                "interval": interval,
                **quality_checks
            }
            
        except Exception as e:
            logger.error(f"Error in data quality summary: {str(e)}")
            return {"status": "error", "message": str(e)}