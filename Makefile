# Tesla Forecasting System Makefile

.PHONY: help install dev test lint format clean up down logs shell migrate

# Default target
help:
	@echo "Tesla Stock Forecasting System"
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  dev        - Run in development mode"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  clean      - Clean up"
	@echo "  up         - Start all services with Docker Compose"
	@echo "  down       - Stop all services"
	@echo "  logs       - Show logs"
	@echo "  shell      - Open shell in backend container"
	@echo "  migrate    - Run database migrations"

# Install dependencies
install:
	pip install -e .

# Development mode
dev:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	pytest backend/tests/ -v --cov=backend

# Linting
lint:
	ruff check backend/ frontend/ orchestration/
	mypy backend/

# Format code
format:
	black backend/ frontend/ orchestration/
	ruff --fix backend/ frontend/ orchestration/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Docker Compose commands
up:
	docker-compose up -d
	@echo "Services starting..."
	@echo "Backend API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/api/v1/docs"
	@echo "Frontend: http://localhost:8501"
	@echo "MLflow: http://localhost:5000"

down:
	docker-compose down

logs:
	docker-compose logs -f

# Open shell in backend container
shell:
	docker-compose exec backend bash

# Database migrations
migrate:
	docker-compose exec backend alembic upgrade head

# Initialize database
init-db:
	docker-compose exec backend python -c "
	from backend.app.models.database import Base
	from backend.app.db.session import sync_engine
	Base.metadata.create_all(bind=sync_engine)
	print('Database initialized')
	"

# Ingest sample data
ingest-data:
	docker-compose exec backend python -c "
	import asyncio
	from backend.app.db.session import AsyncSessionLocal
	from backend.app.services.ingestion import DataIngestionService
	
	async def ingest():
	    async with AsyncSessionLocal() as session:
	        service = DataIngestionService(session)
	        result = await service.fetch_and_store_data('1d', lookback_days=100)
	        print(f'Ingestion result: {result}')
	
	asyncio.run(ingest())
	"

# Compute features
compute-features:
	docker-compose exec backend python -c "
	import asyncio
	from backend.app.db.session import AsyncSessionLocal
	from backend.app.services.features import FeatureEngineeringService
	
	async def compute():
	    async with AsyncSessionLocal() as session:
	        service = FeatureEngineeringService(session)
	        result = await service.compute_and_store_features('1d', lookback_days=100)
	        print(f'Feature computation result: {result}')
	
	asyncio.run(compute())
	"

# Full setup (for first time)
setup: up
	@echo "Waiting for services to start..."
	sleep 10
	make migrate
	make ingest-data
	make compute-features
	@echo "Setup complete! Access the application at:"
	@echo "Frontend: http://localhost:8501"
	@echo "API Docs: http://localhost:8000/api/v1/docs"

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Backend not responding"
	@curl -s http://localhost:8501 > /dev/null && echo "Frontend: OK" || echo "Frontend: Not responding"

# Production build
build:
	docker-compose build --no-cache

# Run specific service
backend:
	docker-compose up backend

frontend:
	docker-compose up frontend

# Backup database
backup:
	docker-compose exec postgres pg_dump -U tesla_user tesla_forecasting > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore database
restore:
	@read -p "Enter backup file name: " file; \
	docker-compose exec -T postgres psql -U tesla_user tesla_forecasting < $$file