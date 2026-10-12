<div align="center">

# 🗄️ `holdon`

**A local-first, single-binary vector database, hybrid search engine, and retrieval tool built in Rust.**

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![SQLite](https://img.shields.io/badge/sqlite-sqlite--vec%20%2B%20fts5-blue.svg?style=flat-square&logo=sqlite)](https://github.com/asg017/sqlite-vec)
[![FastEmbed](https://img.shields.io/badge/embeddings-AllMiniLML6V2%20(384d)-green.svg?style=flat-square)](https://github.com/Anush008/fastembed-rs)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-lightgrey.svg?style=flat-square)](LICENSE)

*Ingest text and PDFs • Semantic Chunking • Dense Vector Search (`sqlite-vec`) • Lexical BM25 (`FTS5`) • Cross-Encoder Reranking (`BGE-Reranker-Base`) • Zero Cloud Dependencies*

</div>

---

## ⚡ Single-Command Quick Install

Install directly to `~/.cargo/bin/holdon` with a single command:

```bash
cargo install --git https://github.com/khokharsnehil45/holdon.git
```

> **Requirement**: Ensure Rust is installed ([`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`](https://rustup.rs)).

Verify installation:
```bash
holdon --version
```

---

## ✨ Why `holdon`?

Most vector databases require heavyweight Docker containers, separate server processes, or cloud API keys with latency and privacy drawbacks.

`holdon` is designed to be **lightweight, embedded, and single-binary**:
1. **Local Embeddings**: Powered by ONNX Runtime (`AllMiniLML6V2`). No OpenAI or third-party API keys required.
2. **Unified SQLite Architecture**: Embeddings (`float[384]`) and text live together in `~/.holdon/holdon.db` powered by `sqlite-vec`.
3. **Multi-Format Ingestion**: Ingests `.txt`, `.md`, `.rs`, `.py`, and complex `.pdf` files out of the box.
4. **Semantic Chunking**: Dynamically splits text on sentence boundaries when semantic similarity drops below a threshold.
5. **Hybrid Search (Dense + Lexical BM25)**: Combines vector similarity with SQLite `FTS5` BM25 keyword matching via **Reciprocal Rank Fusion (RRF)**.
6. **Cross-Encoder Re-ranking**: Optional second-stage reranking via `BGE-Reranker-Base` for state-of-the-art precision.
7. **Local LLM / RAG Ready**: Seamlessly pipes context formatted as clean JSON directly into `ollama`.

---

## 🚀 Quickstart

### 1. Ingest Documents

```bash
# Ingest text or PDF files (sliding window: 512 chars, 50 chars overlap)
holdon --add paper.pdf

# Ingest with Semantic Chunking (splits when topic shifts)
holdon --add notes.txt --semantic --threshold 0.5

# Custom chunk size and overlap
holdon --add book.pdf --chunk-size 600 --overlap 80
```

### 2. Search & Retrieve

```bash
# Basic semantic vector search (top 5 results)
holdon --query "How does memory management work without a GC?" -k 5

# Hybrid Search: Vector + SQLite FTS5 BM25 (ideal for acronyms & model IDs)
holdon --query "DeepSeek-V4.1-Flash parameter count" -k 3 --hybrid

# Maximum Precision: Hybrid Search + Cross-Encoder Reranking
holdon --query "Modality-specific routing biases in MoE" -k 3 --hybrid --rerank

# Output clean JSON for automated pipelines
holdon --query "Quantum entanglement" -k 3 --json
```

---

## 🤖 Local RAG with Ollama (Zero API Keys)

Pipe context retrieved by `holdon` directly into any local Ollama model (e.g. `llama3.2:3b`):

```bash
QUERY="What is the difference between prefill and decode parameter activation?"

(
  echo "You are an assistant answering questions strictly based on the following context."
  echo "--- CONTEXT ---"
  holdon --query "$QUERY" --k 3 --hybrid --rerank --json | jq -r '.[].content'
  echo "--- QUESTION ---"
  echo "$QUERY"
) | ollama run llama3.2:3b
```

### Add a Shell Helper Function (`rag`)

Add this to your `~/.bashrc` or `~/.zshrc`:

```bash
rag() {
  local query="$1"
  local k="${2:-3}"
  (
    echo "Context:"
    holdon --query "$query" --k "$k" --hybrid --rerank --json | jq -r '.[].content'
    echo -e "\nQuestion: $query\nAnswer:"
  ) | ollama run llama3.2:3b
}
```

Then query your documents instantly:
```bash
rag "How are KV cache memory footprint savings achieved in MLA?"
```

---

## 🏗️ Architecture & Retrieval Pipeline

```
                                      User Query
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
           [Dense Vector Search]                      [Lexical BM25 Search]
          (sqlite-vec Cosine Scan)                     (SQLite FTS5 Match)
                      │                                         │
                      ▼                                         ▼
                Dense Ranking                             BM25 Ranking
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
                           [Reciprocal Rank Fusion (RRF)]
                Score(d) = 1/(60 + Rank_dense) + 1/(60 + Rank_bm25)
                                           │
                                           ▼  (with --rerank)
                          [Cross-Encoder Reranker]
                          (BAAI/bge-reranker-base)
                                           │
                                           ▼
                                 Final Top-k Results
```

| Component | Library | Specification |
|---|---|---|
| **CLI & Flags** | `clap` 4.5 | Subcommands (`add`, `query`) and flags (`--add`, `--query`, `--hybrid`, `--rerank`, `-k`, `--k`) |
| **Vector Storage** | `rusqlite` + `sqlite-vec` 0.1.9 | In-process SQLite database with SIMD-accelerated cosine distance |
| **Lexical Search** | SQLite `FTS5` | BM25 full-text search with automatic synchronization triggers |
| **Dense Embeddings** | `fastembed` 7.1.1 | Local `AllMiniLML6V2` ONNX model (384 dimensions) |
| **Reranker** | `fastembed` 7.1.1 | Local Cross-Encoder `BGE-Reranker-Base` |
| **PDF Extraction** | `pdf-extract` 0.12 | Pure Rust document parser and text stream extractor |

---

## 📊 Benchmark Latency Breakdown

Measured on standard x86_64 CPU hardware:

| Step | Latency | Description |
|---|---|---|
| **Vector Search (`sqlite-vec`)** | **~0.4 ms - 0.9 ms** | Sub-millisecond vector cosine scan |
| **Embedding Generation** | **~2.3 ms - 3.4 ms** | Query embedding inference via ONNX Runtime |
| **SQLite Init & WAL** | **~0.4 ms** | Database connection and pragmas |
| **Warm In-Process Query** | **~3.0 ms** | Total time for embedding + search |
| **Cold CLI Process Launch** | **~230 ms** | Spawning fresh binary + loading ONNX model weights into RAM |

---

## 💻 CLI Reference

```
Usage: holdon [COMMAND] [OPTIONS]

Commands:
  add    Ingest a text or PDF file into the vector database
  query  Query the vector database for matching chunks
  help   Print this message or help for the given subcommand

Options:
      --add <FILE_PATH>      Ingest a file (text or PDF)
      --query <QUERY_TEXT>   Search query
      --semantic             Enable semantic chunking based on topic shifts
      --threshold <FLOAT>    Semantic split similarity threshold (default: 0.5)
      --chunk-size <CHARS>   Chunk character length (default: 512)
      --overlap <CHARS>      Chunk overlap characters (default: 50)
  -k, --k <TOP_K>            Number of results to retrieve (default: 5)
      --hybrid               Enable Hybrid Search (Dense Vectors + FTS5 BM25 via RRF)
      --rerank               Enable Cross-Encoder Reranking (BGE-Reranker-Base)
      --candidates <N>       Candidate pool retrieved before reranking (default: 20)
      --json                 Format output as JSON
  -h, --help                 Print help
  -V, --version              Print version
```

---

## 🛠️ Building from Source

```bash
# Clone the repository
git clone https://github.com/khokharsnehil45/holdon.git
cd holdon

# Run test suite
cargo test

# Build optimized release binary
cargo build --release

# Binary is available at target/release/holdon
```

---

## 📜 License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](http://www.apache.org/licenses/LICENSE-2.0))
- MIT license ([LICENSE-MIT](http://opensource.org/licenses/MIT))

at your option.
