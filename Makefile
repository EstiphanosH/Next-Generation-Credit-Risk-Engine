# Build automation
.PHONY: test lint format

test:
	pytest -v tests/

lint:
	flake8 scripts/ tests/ api/

format:
	black scripts/ tests/ api/
