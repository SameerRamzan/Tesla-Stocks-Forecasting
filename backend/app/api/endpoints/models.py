"""Model management endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.db.session import get_async_session
from backend.app.models.database import ModelRegistry
from backend.app.schemas.api import ModelListResponse, ModelInfo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    session: AsyncSession = Depends(get_async_session)
):
    """List all registered models."""
    try:
        query = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
        result = await session.execute(query)
        records = result.scalars().all()
        
        models = [
            ModelInfo(
                model_id=record.model_id,
                name=record.name,
                framework=record.framework,
                version=record.version,
                stage=record.stage,
                active=record.active,
                created_at=record.created_at,
                registered_at=record.registered_at,
                cv_mae=record.cv_mae,
                cv_rmse=record.cv_rmse,
                cv_mape=record.cv_mape,
                cv_directional_accuracy=record.cv_directional_accuracy,
                training_start=record.training_start,
                training_end=record.training_end,
                features_used=record.features_used,
                target_variable=record.target_variable
            )
            for record in records
        ]
        
        return ModelListResponse(models=models, count=len(models))
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/production")
async def get_production_model(
    session: AsyncSession = Depends(get_async_session)
):
    """Get the current production model."""
    try:
        query = select(ModelRegistry).where(
            ModelRegistry.stage == "Production",
            ModelRegistry.active == True
        ).order_by(ModelRegistry.registered_at.desc()).limit(1)
        
        result = await session.execute(query)
        record = result.scalar()
        
        if not record:
            raise HTTPException(status_code=404, detail="No production model found")
        
        return ModelInfo(
            model_id=record.model_id,
            name=record.name,
            framework=record.framework,
            version=record.version,
            stage=record.stage,
            active=record.active,
            created_at=record.created_at,
            registered_at=record.registered_at,
            cv_mae=record.cv_mae,
            cv_rmse=record.cv_rmse,
            cv_mape=record.cv_mape,
            cv_directional_accuracy=record.cv_directional_accuracy,
            training_start=record.training_start,
            training_end=record.training_end,
            features_used=record.features_used,
            target_variable=record.target_variable
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting production model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))