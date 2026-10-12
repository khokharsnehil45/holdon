# holdon

A production-grade, local-first vector database, hybrid search engine, and retrieval CLI written in Rust.

`holdon` ingests text and PDF documents, performs semantic or sliding-window chunking, computes local embeddings using ONNX (`AllMiniLML6V2`), stores vectors in a unified SQLite database via `sqlite-vec` + `FTS5`, and executes high-precision retrieval using Hybrid Search (BM25 + Vector Reciprocal Rank Fusion) and Cross-Encoder Re-ranking (`BGE-Reranker-Base`).

---

## Features

- **Local-First & Zero-Cloud**: No API keys, subscriptions, or external network requests during inference.
- **Embedded Storage**: Single SQLite database file (`~/.holdon/holdon.db`) with native SIMD-accelerated `sqlite-vec` integration.
- **Full PDF & Text Ingestion**: Handles raw `.txt`, `.md`, `.rs`, and `.pdf` documents directly.
- **Semantic Chunking**: Splits text along sentence boundaries when semantic similarity drops below a configurable threshold.
- **Hybrid Search (Dense + Lexical BM25)**: Blends vector embeddings with SQLite FTS5 keyword matching using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-ranking**: Optional 2nd-stage scoring via `BGE-Reranker-Base` for state-of-the-art precision.
- **First-Class CLI & Flags**: Supports subcommands (`add`, `query`) and flags (`--add`, `--query`, `--k`, `-k`, `--json`).
- **Ollama / Local RAG Ready**: Clean JSON output mode (`--json`) formatted for direct piping into local LLMs.

---

## Installation

Ensure you have Rust and Cargo installed:

```bash
git clone https://github.com/khokharsnehil45/holdon.git
cd holdon
cargo install --path .
```

`holdon` will be installed to `~/.cargo/bin/holdon`.

---

## Usage

### 1. Ingestion (`add` / `--add`)

```bash
# Ingest with default sliding window (512 chars, 50 chars overlap)
holdon --add paper.pdf

# Ingest using Semantic Chunking (cosine similarity threshold = 0.5)
holdon --add research.pdf --semantic --threshold 0.5

# Custom chunk size and overlap
holdon add notes.txt --chunk-size 600 --overlap 80
```

### 2. Retrieval (`query` / `--query`)

```bash
# Basic vector search (top 5 matches)
holdon --query "How does memory management work?" -k 5

# Hybrid Search (Dense Vectors + SQLite FTS5 BM25 with Reciprocal Rank Fusion)
holdon --query "DeepSeek-V4.1-Flash parameters" -k 3 --hybrid

# Maximum Precision: Hybrid Search + Cross-Encoder Re-ranking
holdon --query "Auxiliary-loss load balancing in MoE" -k 3 --hybrid --rerank

# JSON Output for downstream pipelines
holdon --query "Quantum entanglement" -k 3 --json
```

---

## Local RAG with Ollama

Pipe retrieved context directly into a local model (e.g. `llama3.2:3b`):

```bash
QUERY="What is the difference between prefill and decode parameter activation?"

(
  echo "You are an assistant answering strictly based on the provided context:"
  echo "--- CONTEXT ---"
  holdon --query "$QUERY" --k 3 --hybrid --rerank --json | jq -r '.[].content'
  echo "--- QUESTION ---"
  echo "$QUERY"
) | ollama run llama3.2:3b
```

---

## Architecture

| Component | Technology | Description |
|---|---|---|
| **CLI Framework** | `clap` 4.5 | Derive syntax with subcommands and interchangeable flags |
| **Database** | `rusqlite` + `sqlite-vec` | Embedded SQLite with SIMD vector extension & WAL mode |
| **Full-Text Search** | SQLite `FTS5` | BM25 lexical keyword matching with automatic sync triggers |
| **Embeddings** | `fastembed` 7.1 | Local ONNX `AllMiniLML6V2` (384 dimensions) |
| **Reranker** | `fastembed` 7.1 | Local Cross-Encoder `BGE-Reranker-Base` |
| **PDF Extraction** | `pdf-extract` 0.12 | Native text extraction from PDF streams |
| **Serialization** | `serde` + `serde_json` | JSON output and vector byte casting |

---

## License

MIT OR Apache-2.0
