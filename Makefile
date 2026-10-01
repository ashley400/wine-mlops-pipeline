.PHONY: install lint test train evaluate clean

install:
	python -m pip install -r requirements.txt

lint:
	python -m flake8 src tests

test:
	python -m pytest -q

train:
	python -m src.train

evaluate:
	python -m src.evaluate

clean:
	python -c "import shutil; shutil.rmtree('mlruns', ignore_errors=True); shutil.rmtree('mlartifacts', ignore_errors=True); shutil.rmtree('results', ignore_errors=True); shutil.rmtree('.pytest_cache', ignore_errors=True); shutil.rmtree('__pycache__', ignore_errors=True); shutil.rmtree('src/__pycache__', ignore_errors=True); shutil.rmtree('tests/__pycache__', ignore_errors=True)"