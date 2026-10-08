# Contributing to Production-Grade RAG System

Thank you for your interest in contributing to this project!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/prosuvinhere/Production-Grade-Retrieval-Augmented-Generation-RAG-System.git
   cd Production-Grade-Retrieval-Augmented-Generation-RAG-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Setup environment variables:**
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY and COHERE_API_KEY
   ```

## Running Tests

Run the test suite prior to opening a Pull Request:
```bash
pytest -v
```

## Running Evaluation Gate

To test Ragas offline faithfulness locally:
```bash
python main.py ingest --dir data/sample_docs
python src/evaluation/evaluate.py
```

## Pull Request Guidelines

1. Create a descriptive feature branch (`git checkout -b feat/your-feature-name`).
2. Adhere to Python code style and formatting standards.
3. Add unit tests for any new modules or algorithms.
4. Ensure prompts preserve strict citation rules.
5. Submit your PR using the provided PR template.
