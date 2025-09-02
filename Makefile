VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL := sentence-transformers/all-MiniLM-L6-v2
PDF_DIR := data
TXT_DIR := data

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

process-pdf:
	@echo "🔍 Обработка PDF-файлов..."
	@$(PYTHON) scripts/pdf_to_txt.py $(PDF_DIR)

check-quality:
	@echo "📊 Проверка качества текстов..."
	@$(PYTHON) scripts/check_quality.py $(TXT_DIR)

build-index:
	@echo "⚙️ Построение индекса..."
	@PYTHONPATH=$(PWD) $(PYTHON) -m scripts.build_index --model $(MODEL) --data_dir $(TXT_DIR) --index_type ip

reindex:
	@echo "🔄 Переиндексация..."
	@PYTHONPATH=$(PWD) $(PYTHON) -m scripts.build_index --model $(MODEL) --data_dir $(TXT_DIR) --index_type ip

all: install process-pdf check-quality build-index
