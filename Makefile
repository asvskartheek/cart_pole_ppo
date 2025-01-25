.PHONY: install test lint format clean

install:
	pip install -r requirements-dev.txt

test:
	pytest tests/ --cov=./ --cov-report=term-missing

lint:
	flake8 .
	mypy ppo.py
	black --check .
	isort --check-only .

format:
	black .
	isort .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +