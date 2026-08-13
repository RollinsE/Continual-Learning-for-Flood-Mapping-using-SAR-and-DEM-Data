.PHONY: install install-colab test smoke build clean

install:
	python -m pip install -r requirements.txt -r requirements-dev.txt
	python -m pip install -e .

install-colab:
	python -m pip install -r requirements-colab.txt
	python -m pip install --no-deps --no-build-isolation -e .

test:
	python -m compileall -q floods scripts streamlit_app tests
	python -m pytest -q

smoke:
	python scripts/runtime_smoke.py

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
