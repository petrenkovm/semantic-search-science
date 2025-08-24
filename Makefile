VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL := sentence-transformers/all-MiniLM-L6-v2

install:
		python3 -m venv $(VENV)
		$(PIP) install --upgrade pip
		$(PIP) install -r requirements.txt

build-index:
		PYTHONPATH=$(PWD) $(PYTHON) -m scripts.build_index --model $(MODEL)

all: install build-index
