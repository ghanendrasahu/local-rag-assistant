# Local RAG Assistant: Interview-Ready Project

Local RAG Assistant is a production-style Retrieval-Augmented Generation (RAG) starter project that answers questions over local `.txt` and `.pdf` files.

It is built to be:
- Easy to demo in interviews
- Easy to extend into advanced RAG patterns
- Fully local for ingestion, chunking, indexing, and retrieval

## What This Project Includes

- Clean `src/` package layout
- Config-driven RAG pipeline
- Document ingestion for TXT and PDF
- Overlapping text chunking strategy
- TF-IDF retrieval index with persistence
- CLI commands for indexing and querying
- Unit tests with `pytest`
- Sample dataset under `data/text_files/`

## Project Structure

```text
local-rag-assistant/
  data/
    text_files/
  src/
    local_rag_assistant/
      cli.py
      config.py
      ingest.py
      chunking.py
      index.py
      pipeline.py
  tests/
    test_chunking.py
    test_pipeline.py
  main.py
  pyproject.toml
```

## Quick Start

1. Create environment and install dependencies.
2. Run indexing:
```bash
python main.py index
```
3. Ask a question:
```bash
python main.py ask --query "What is Python used for?"
```
4. Run a full demo:
```bash
python main.py demo
```

## CLI Commands

- `index`: Build a retrieval index from `data/`.
- `ask --query "..."`: Query the existing index and return answer + source scores.
- `demo`: Index and run sample questions.

Useful flags:
- `--data-dir` (default: `data`)
- `--index-dir` (default: `artifacts/index`)
- `--chunk-size` (default: `700`)
- `--chunk-overlap` (default: `120`)
- `--top-k` (default: `4`)

## Testing

```bash
python -m pytest
```

## Interview Talking Points

- Why chunk overlap improves recall near boundaries.
- Why TF-IDF is a strong baseline before moving to embeddings/vector DB.
- How to measure retrieval quality (recall@k, MRR, answer groundedness).
- How to productionize next:
  - Add dense embeddings and vector store (FAISS/Chroma).
  - Add LLM answer synthesis with citation enforcement.
  - Add evaluation harness and regression tests for RAG quality.

## Roadmap

- Add hybrid retrieval (keyword + dense)
- Add source-aware answer citations
- Add API layer (FastAPI)
- Add Docker + CI workflow
