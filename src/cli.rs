use std::path::PathBuf;
use clap::{ArgGroup, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "holdon",
    version = "1.0.0",
    about = "Local-first vector & hybrid database retrieval tool",
    long_about = "A local-first, single-binary vector database and retrieval tool that ingests text and PDF files, chunks them, generates local embeddings, stores them in a unified SQLite database, and runs dense vector, hybrid (BM25 + Vector RRF), and cross-encoder similarity searches."
)]
#[command(group(
    ArgGroup::new("mode")
        .args(["add", "query"])
        .conflicts_with("subcommand")
))]
pub struct Cli {
    /// Ingest a text or PDF file into the vector database (flag mode)
    #[arg(long, value_name = "FILE_PATH")]
    pub add: Option<PathBuf>,

    /// Query the database for matching chunks (flag mode)
    #[arg(long, value_name = "QUERY_TEXT")]
    pub query: Option<String>,

    /// Enable semantic chunking (splits on sentence boundaries when meaning shifts)
    #[arg(long)]
    pub semantic: bool,

    /// Semantic similarity split threshold (default: 0.5; range: 0.0 - 1.0)
    #[arg(long, default_value_t = 0.5, value_name = "FLOAT")]
    pub threshold: f32,

    /// Chunk character length for fixed chunking, or max chars for semantic (default: 512)
    #[arg(long, default_value_t = 512, value_name = "CHARS")]
    pub chunk_size: usize,

    /// Overlap character count between consecutive chunks in fixed mode (default: 50)
    #[arg(long, default_value_t = 50, value_name = "CHARS")]
    pub overlap: usize,

    /// Number of worker threads for parallel chunking (default: 0 = all CPU cores)
    #[arg(long, default_value_t = 0, value_name = "THREADS")]
    pub threads: usize,

    /// Number of top similar chunks to return (supports -k, --k, --top-k; default: 5)
    #[arg(short = 'k', long = "k", alias = "top-k", default_value_t = 5, value_name = "TOP_K")]
    pub k: usize,

    /// Enable Hybrid Search (Vector + SQLite FTS5 BM25 combined via Reciprocal Rank Fusion)
    #[arg(long)]
    pub hybrid: bool,

    /// Enable Cross-Encoder Reranking (BGE-Reranker-Base) for highest accuracy
    #[arg(long)]
    pub rerank: bool,

    /// Number of candidate chunks retrieved before reranking or hybrid blending (default: 20)
    #[arg(long, default_value_t = 20, value_name = "CANDIDATES")]
    pub candidates: usize,

    /// Output the results formatted as JSON
    #[arg(long)]
    pub json: bool,

    /// Optional subcommand mode: `holdon add ...` or `holdon query ...`
    #[command(subcommand)]
    pub subcommand: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Ingest a text or PDF file into the vector database
    Add {
        /// Path to the text or PDF file to ingest
        #[arg(value_name = "FILE_PATH")]
        file_path: PathBuf,

        /// Enable semantic chunking based on sentence embedding divergence
        #[arg(long)]
        semantic: bool,

        /// Semantic similarity split threshold (default: 0.5)
        #[arg(long, default_value_t = 0.5, value_name = "FLOAT")]
        threshold: f32,

        /// Chunk character length (default: 512)
        #[arg(long, default_value_t = 512, value_name = "CHARS")]
        chunk_size: usize,

        /// Overlap character count between consecutive chunks (default: 50)
        #[arg(long, default_value_t = 50, value_name = "CHARS")]
        overlap: usize,

        /// Number of worker threads for parallel chunking (default: 0 = all CPU cores)
        #[arg(long, default_value_t = 0, value_name = "THREADS")]
        threads: usize,
    },

    /// Query the vector database for semantically similar chunks
    Query {
        /// Natural language query text
        #[arg(value_name = "QUERY_TEXT")]
        query_text: String,

        /// Number of top similar chunks to return (supports -k, --k, --top-k; default: 5)
        #[arg(short = 'k', long = "k", alias = "top-k", default_value_t = 5, value_name = "TOP_K")]
        top_k: usize,

        /// Enable Hybrid Search (Vector + FTS5 BM25 via Reciprocal Rank Fusion)
        #[arg(long)]
        hybrid: bool,

        /// Enable Cross-Encoder Reranking (BGE-Reranker-Base) for highest accuracy
        #[arg(long)]
        rerank: bool,

        /// Number of candidate chunks retrieved before reranking or blending (default: 20)
        #[arg(long, default_value_t = 20, value_name = "CANDIDATES")]
        candidates: usize,

        /// Output the results formatted as JSON
        #[arg(long)]
        json: bool,
    },
}

/// Normalized action resolved from either top-level flags or subcommands.
pub enum Action {
    Add {
        file_path: PathBuf,
        semantic: bool,
        threshold: f32,
        chunk_size: usize,
        overlap: usize,
        threads: usize,
    },
    Query {
        query_text: String,
        top_k: usize,
        hybrid: bool,
        rerank: bool,
        candidates: usize,
        json: bool,
    },
}

impl Cli {
    pub fn resolve_action(self) -> Result<Action, String> {
        if let Some(cmd) = self.subcommand {
            match cmd {
                Commands::Add {
                    file_path,
                    semantic,
                    threshold,
                    chunk_size,
                    overlap,
                    threads,
                } => Ok(Action::Add {
                    file_path,
                    semantic,
                    threshold,
                    chunk_size,
                    overlap,
                    threads,
                }),
                Commands::Query {
                    query_text,
                    top_k,
                    hybrid,
                    rerank,
                    candidates,
                    json,
                } => Ok(Action::Query {
                    query_text,
                    top_k,
                    hybrid,
                    rerank,
                    candidates,
                    json,
                }),
            }
        } else if let Some(file_path) = self.add {
            Ok(Action::Add {
                file_path,
                semantic: self.semantic,
                threshold: self.threshold,
                chunk_size: self.chunk_size,
                overlap: self.overlap,
                threads: self.threads,
            })
        } else if let Some(query_text) = self.query {
            Ok(Action::Query {
                query_text,
                top_k: self.k,
                hybrid: self.hybrid,
                rerank: self.rerank,
                candidates: self.candidates,
                json: self.json,
            })
        } else {
            Err("No action specified. Run with `--add <file>`, `--query <text>`, or use subcommands (`holdon --help`).".to_string())
        }
    }
}
