format:
	black .
	isort .

test:
	black .
	isort .
	env PYTHONPATH=. pytest --pylint --flake8 --mypy
