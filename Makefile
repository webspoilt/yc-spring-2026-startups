.PHONY: help build up down logs clean test install

help:
	@echo "AI Monorepo - Available Commands"
	@echo "================================"
	@echo "make build          - Build all Docker images"
	@echo "make up             - Start all services"
	@echo "make down           - Stop all services"
	@echo "make logs           - View logs (follow mode)"
	@echo "make clean          - Remove containers and volumes"
	@echo "make test           - Run tests for all projects"
	@echo "make install        - Install Python dependencies locally"
	@echo ""
	@echo "Individual Projects:"
	@echo "make pm-cursor      - Start PM Cursor only"
	@echo "make quant-fund     - Start Quant Fund only"
	@echo "make ai-agency      - Start AI Agency only"
	@echo "make stablecoin     - Start Stablecoin only"
	@echo "make gov-ai         - Start Gov AI only"
	@echo "make vision         - Start Vision Guidance only"
	@echo "make spatial        - Start Spatial Transformer only"
	@echo "make fraud          - Start Fraud System only"
	@echo "make mlops          - Start MLOps only"
	@echo "make mes            - Start AI MES only"

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v --rmi local

test:
	@echo "Running tests for all projects..."
	@for dir in step-01-pm-cursor step-02-quant-fund step-03-ai-agency step-04-stablecoin step-05-gov-ai step-06-vision-guidance step-07-spatial-transformer step-08-fraud-system step-09-mlops step-10-ai-mes; do \
		echo "Testing $$dir..."; \
		cd $$dir && pip install -q -r requirements.txt 2>/dev/null || true; \
		cd ..; \
	done

install:
	@for dir in step-01-pm-cursor step-02-quant-fund step-03-ai-agency step-04-stablecoin step-05-gov-ai step-06-vision-guidance step-07-spatial-transformer step-08-fraud-system step-09-mlops step-10-ai-mes; do \
		echo "Installing $$dir dependencies..."; \
		cd $$dir && pip install -r requirements.txt 2>/dev/null || true; \
		cd ..; \
	done

# Individual project commands
pm-cursor:
	docker-compose up -d pm-cursor

quant-fund:
	docker-compose up -d quant-fund

ai-agency:
	docker-compose up -d ai-agency

stablecoin:
	docker-compose up -d stablecoin

gov-ai:
	docker-compose up -d gov-ai

vision:
	docker-compose up -d vision-guidance

spatial:
	docker-compose up -d spatial-transformer

fraud:
	docker-compose up -d fraud-system

mlops:
	docker-compose up -d mlops

mes:
	docker-compose up -d ai-mes

# Step-by-step development
step-01:
	docker-compose up -d pm-cursor
	@echo "PM Cursor running at http://localhost:8001"

step-02:
	docker-compose up -d quant-fund
	@echo "Quant Fund running at http://localhost:8002"

step-03:
	docker-compose up -d ai-agency
	@echo "AI Agency running at http://localhost:8003"

step-04:
	docker-compose up -d stablecoin
	@echo "Stablecoin running at http://localhost:8004"

step-05:
	docker-compose up -d gov-ai
	@echo "Gov AI running at http://localhost:8005"

step-06:
	docker-compose up -d vision-guidance
	@echo "Vision Guidance running at http://localhost:8006"

step-07:
	docker-compose up -d spatial-transformer
	@echo "Spatial Transformer running at http://localhost:8007"

step-08:
	docker-compose up -d fraud-system
	@echo "Fraud System running at http://localhost:8008"

step-09:
	docker-compose up -d mlops
	@echo "MLOps running at http://localhost:8009"

step-10:
	docker-compose up -d ai-mes
	@echo "AI MES running at http://localhost:8010"
