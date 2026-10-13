use std::time::Instant;
use anyhow::{bail, Context, Result};
use clap::Parser;

use holdon::cli::{Action, Cli};
use holdon::chunker::{chunk_text_fixed, chunk_text_semantic};
use holdon::db::Database;
use holdon::embedder::Embedder;
use holdon::reader::read_document_text;
use holdon::reranker::Reranker;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let action = match cli.resolve_action() {
        Ok(a) => a,
        Err(err_msg) => {
            eprintln!("Error: {}\n", err_msg);
            eprintln!("Usage examples:");
            eprintln!("  holdon --add doc.pdf --semantic");
            eprintln!("  holdon --add doc.pdf --threads 8");
            eprintln!("  holdon --query \"search query\" --k 3 --hybrid");
            eprintln!("  holdon --query \"search query\" --k 3 --hybrid --rerank");
            eprintln!("  holdon query \"search query\" -k 3 --hybrid --json");
            bail!("Missing required arguments");
        }
    };

    match action {
        Action::Add {
            file_path,
            semantic,
            threshold,
            chunk_size,
            overlap,
            threads,
        } => {
            let start_time = Instant::now();

            // Configure Rayon thread pool if custom thread count requested
            if threads > 0 {
                let _ = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build_global();
            }

            let canonical_path = file_path
                .canonicalize()
                .with_context(|| format!("Failed to resolve file path: {}", file_path.display()))?;

            let text_content = read_document_text(&canonical_path)?;

            // Initialize embedder
            let mut embedder = Embedder::try_new()
                .context("Failed to initialize FastEmbed embedding model")?;

            let chunks = if semantic {
                chunk_text_semantic(
                    &text_content,
                    &mut embedder,
                    threshold,
                    chunk_size,       // max chunk chars
                    64,               // min chunk chars before split
                ).with_context(|| "Failed to execute semantic chunking on document".to_string())?
            } else {
                chunk_text_fixed(&text_content, chunk_size, overlap).with_context(|| {
                    format!(
                        "Failed to chunk text with chunk_size={} and overlap={}",
                        chunk_size, overlap
                    )
                })?
            };

            if chunks.is_empty() {
                println!(
                    "Warning: File '{}' contains no indexable text chunks.",
                    canonical_path.display()
                );
                return Ok(());
            }

            let file_str = canonical_path.to_string_lossy().to_string();

            // Batch embed the final chunks
            let text_slices: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
            let embeddings = embedder
                .embed_batch(&text_slices)
                .context("Failed to generate embeddings for text chunks")?;

            // Insert into unified database (both chunks vector table and chunks_fts)
            let db_path = Database::default_db_path()
                .context("Failed to resolve default database path (~/.holdon/holdon.db)")?;
            let mut db = Database::open(&db_path)
                .with_context(|| format!("Failed to open database at '{}'", db_path.display()))?;

            let total_chunks = db
                .insert_chunks(&file_str, &chunks, &embeddings)
                .context("Failed to insert chunk records into database transaction")?;

            let elapsed = start_time.elapsed();
            let active_threads = if threads > 0 { threads } else { rayon::current_num_threads() };

            println!("Successfully ingested file:");
            println!("  Source file:    {}", file_str);
            println!("  Mode:           {}", if semantic { format!("Parallel Semantic (threshold: {:.2})", threshold) } else { "Parallel Fixed sliding-window".to_string() });
            println!("  Worker threads: {}", active_threads);
            println!("  Total chunks:   {}", total_chunks);
            println!("  Time taken:     {:.2?}", elapsed);
        }

        Action::Query {
            query_text,
            top_k,
            hybrid,
            rerank,
            candidates,
            json,
        } => {
            if top_k == 0 {
                if json {
                    println!("[]");
                } else {
                    println!("No results returned (k=0).");
                }
                return Ok(());
            }

            let db_path = Database::default_db_path()
                .context("Failed to resolve default database path (~/.holdon/holdon.db)")?;

            if !db_path.exists() {
                if json {
                    println!("[]");
                } else {
                    println!(
                        "Database not found at '{}'. Run `holdon --add <file>` first to index files.",
                        db_path.display()
                    );
                }
                return Ok(());
            }

            let db = Database::open(&db_path)
                .with_context(|| format!("Failed to open database at '{}'", db_path.display()))?;

            let mut embedder = Embedder::try_new()
                .context("Failed to initialize FastEmbed embedding model")?;
            let query_embedding = embedder
                .embed_query(&query_text)
                .context("Failed to generate embedding for query")?;

            // Candidate pool size when hybrid search or reranker is used
            let candidate_k = if rerank || hybrid {
                top_k.max(candidates)
            } else {
                top_k
            };

            // 1. Initial retrieval: Hybrid (FTS5 + Vector RRF) or pure Vector
            let initial_results = if hybrid {
                db.query_hybrid(&query_text, &query_embedding, if rerank { candidate_k } else { top_k }, candidate_k)
                    .context("Failed to perform hybrid vector + FTS search")?
            } else {
                db.query_top_k(&query_embedding, candidate_k)
                    .context("Failed to perform vector similarity query")?
            };

            // 2. Second stage: Cross-encoder reranker if enabled
            let results = if rerank && !initial_results.is_empty() {
                let mut reranker = Reranker::try_new()
                    .context("Failed to initialize Cross-Encoder Reranker model")?;
                reranker
                    .rerank(&query_text, initial_results, top_k)
                    .context("Failed to execute cross-encoder reranking")?
            } else if !rerank && !hybrid {
                initial_results.into_iter().take(top_k).collect()
            } else {
                initial_results
            };

            if json {
                let json_output = serde_json::to_string_pretty(&results)
                    .context("Failed to serialize results to JSON")?;
                println!("{}", json_output);
            } else if results.is_empty() {
                println!("No matching chunks found in database.");
            } else {
                let mode_str = match (hybrid, rerank) {
                    (true, true) => " [Hybrid BM25+Vector -> Cross-Encoder Reranked]",
                    (true, false) => " [Hybrid BM25+Vector RRF]",
                    (false, true) => " [Vector Search -> Cross-Encoder Reranked]",
                    (false, false) => " [Vector Search]",
                };
                println!(
                    "Found {} results for query: \"{}\"{}\n",
                    results.len(),
                    query_text,
                    mode_str
                );
                for (rank, res) in results.iter().enumerate() {
                    println!(
                        "[{}] Score: {:.4} | {}:{}",
                        rank + 1,
                        res.score,
                        res.source_file,
                        res.chunk_index
                    );
                    println!("    Content: {}\n", res.content.trim());
                }
            }
        }
    }

    Ok(())
}
