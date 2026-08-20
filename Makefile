.PHONY: test coverage

test:
	python3 -m pytest

coverage:
	python3 -m pytest --cov=. --cov-report=term-missing
