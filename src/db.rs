use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use rusqlite::ffi::sqlite3_auto_extension;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::chunker::TextChunk;
use crate::error::HoldonError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: i64,
    pub score: f64,
    pub source_file: String,
    pub chunk_index: usize,
    pub content: String,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Return the canonical database path at `~/.holdon/holdon.db`.
    pub fn default_db_path() -> Result<PathBuf, HoldonError> {
        let home_dir = dirs::home_dir()
            .ok_or_else(|| HoldonError::Config("Could not locate user home directory".to_string()))?;
        Ok(home_dir.join(".holdon").join("holdon.db"))
    }

    /// Open or create the SQLite database with sqlite-vec extension auto-registered.
    pub fn open(path: &Path) -> Result<Self, HoldonError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| HoldonError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        // Register the sqlite-vec extension globally for sqlite3 connections
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }

        let conn = Connection::open(path)?;

        // Performance pragmas
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;",
        )?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Open in-memory database for testing.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self, HoldonError> {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initialize the unified `chunks` table and FTS5 full-text index if they do not already exist.
    fn init_schema(&self) -> Result<(), HoldonError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding float[384],
                content TEXT NOT NULL,
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);

            -- FTS5 full-text index for lexical keyword search
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            );

            -- Triggers to automatically keep FTS5 synchronized with chunks
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.id, old.content);
                INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;",
        )?;
        Ok(())
    }

    /// Insert chunks with their corresponding 384-dimensional embeddings within a single transaction.
    pub fn insert_chunks(
        &mut self,
        source_file: &str,
        chunks: &[TextChunk],
        embeddings: &[Vec<f32>],
    ) -> Result<usize, HoldonError> {
        if chunks.len() != embeddings.len() {
            return Err(HoldonError::InvalidInput(format!(
                "Mismatch between chunk count ({}) and embedding count ({})",
                chunks.len(),
                embeddings.len()
            )));
        }

        if chunks.is_empty() {
            return Ok(0);
        }

        let tx = self.conn.transaction()?;

        {
            let mut stmt = tx.prepare(
                "INSERT INTO chunks (embedding, content, source_file, chunk_index)
                 VALUES (?, ?, ?, ?)",
            )?;

            for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
                let emb_bytes: &[u8] = bytemuck::cast_slice(embedding.as_slice());
                stmt.execute(params![
                    emb_bytes,
                    &chunk.content,
                    source_file,
                    chunk.chunk_index as i64
                ])?;
            }
        }

        tx.commit()?;
        Ok(chunks.len())
    }

    /// Perform a dense top-k cosine similarity search using `vec_distance_cosine`.
    pub fn query_top_k(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, HoldonError> {
        let query_bytes: &[u8] = bytemuck::cast_slice(query_embedding);

        let mut stmt = self.conn.prepare(
            "SELECT id,
                    vec_distance_cosine(embedding, ?) AS distance,
                    source_file,
                    chunk_index,
                    content
             FROM chunks
             WHERE embedding IS NOT NULL
             ORDER BY distance ASC
             LIMIT ?",
        )?;

        let rows = stmt.query_map(params![query_bytes, top_k as i64], |row| {
            let id: i64 = row.get(0)?;
            let distance: f64 = row.get(1)?;
            let source_file: String = row.get(2)?;
            let chunk_index: i64 = row.get(3)?;
            let content: String = row.get(4)?;

            let score = 1.0 - distance;

            Ok(SearchResult {
                id,
                score,
                source_file,
                chunk_index: chunk_index as usize,
                content,
            })
        })?;

        let mut results = Vec::new();
        for r in rows {
            results.push(r?);
        }

        Ok(results)
    }

    /// Clean query text for FTS5 syntax safety.
    fn sanitize_fts5_query(query: &str) -> String {
        let words: Vec<&str> = query
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return String::new();
        }

        // Format as prefix queries joined by OR
        words
            .iter()
            .map(|w| format!("\"{}\"*", w))
            .collect::<Vec<_>>()
            .join(" OR ")
    }

    /// Perform a lexical BM25 keyword search using SQLite FTS5.
    pub fn query_fts_top_k(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, HoldonError> {
        let fts_query = Self::sanitize_fts5_query(query_text);
        if fts_query.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = self.conn.prepare(
            "SELECT c.id,
                    f.rank,
                    c.source_file,
                    c.chunk_index,
                    c.content
             FROM chunks_fts f
             JOIN chunks c ON c.id = f.rowid
             WHERE chunks_fts MATCH ?
             ORDER BY f.rank ASC
             LIMIT ?",
        )?;

        let rows = stmt.query_map(params![fts_query, top_k as i64], |row| {
            let id: i64 = row.get(0)?;
            let bm25_rank: f64 = row.get(1)?;
            let source_file: String = row.get(2)?;
            let chunk_index: i64 = row.get(3)?;
            let content: String = row.get(4)?;

            // Convert negative bm25 score to positive rank score
            let score = (-bm25_rank).max(0.0);

            Ok(SearchResult {
                id,
                score,
                source_file,
                chunk_index: chunk_index as usize,
                content,
            })
        });

        match rows {
            Ok(mapped_rows) => {
                let mut results = Vec::new();
                for r in mapped_rows {
                    results.push(r?);
                }
                Ok(results)
            }
            Err(_) => Ok(Vec::new()), // Fallback gracefully if query is invalid syntax
        }
    }

    /// Hybrid Search: Combines Dense Vector Search + FTS5 BM25 Keyword Search
    /// using Reciprocal Rank Fusion (RRF):
    ///
    /// RRF_score(d) = (1 / (rrf_k + rank_dense)) + (1 / (rrf_k + rank_fts))
    pub fn query_hybrid(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        candidate_pool: usize,
    ) -> Result<Vec<SearchResult>, HoldonError> {
        let pool_size = top_k.max(candidate_pool);

        // 1. Fetch dense candidates
        let dense_results = self.query_top_k(query_embedding, pool_size)?;

        // 2. Fetch lexical BM25 candidates
        let fts_results = self.query_fts_top_k(query_text, pool_size)?;

        if fts_results.is_empty() {
            return Ok(dense_results.into_iter().take(top_k).collect());
        }
        if dense_results.is_empty() {
            return Ok(fts_results.into_iter().take(top_k).collect());
        }

        const RRF_K: f64 = 60.0;
        let mut rrf_scores: HashMap<i64, f64> = HashMap::new();
        let mut doc_map: HashMap<i64, SearchResult> = HashMap::new();

        // Accumulate dense ranks
        for (rank, item) in dense_results.into_iter().enumerate() {
            let score = 1.0 / (RRF_K + (rank + 1) as f64);
            *rrf_scores.entry(item.id).or_insert(0.0) += score;
            doc_map.insert(item.id, item);
        }

        // Accumulate lexical BM25 ranks
        for (rank, item) in fts_results.into_iter().enumerate() {
            let score = 1.0 / (RRF_K + (rank + 1) as f64);
            *rrf_scores.entry(item.id).or_insert(0.0) += score;
            doc_map.entry(item.id).or_insert(item);
        }

        // Sort by blended RRF score descending
        let mut scored_results: Vec<(i64, f64)> = rrf_scores.into_iter().collect();
        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut final_results = Vec::with_capacity(top_k);
        for (id, rrf_score) in scored_results.into_iter().take(top_k) {
            if let Some(mut doc) = doc_map.remove(&id) {
                doc.score = rrf_score;
                final_results.push(doc);
            }
        }

        Ok(final_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_db_insert_and_query() {
        let mut db = Database::open_in_memory().unwrap();

        let chunks = vec![
            TextChunk {
                content: "Rust systems programming".to_string(),
                chunk_index: 0,
            },
            TextChunk {
                content: "Baking sourdough bread".to_string(),
                chunk_index: 1,
            },
        ];

        let mut v1 = vec![0.0f32; 384];
        v1[0] = 1.0;
        let mut v2 = vec![0.0f32; 384];
        v2[1] = 1.0;

        let embeddings = vec![v1.clone(), v2.clone()];
        let inserted = db.insert_chunks("doc.txt", &chunks, &embeddings).unwrap();
        assert_eq!(inserted, 2);

        // Query closer to v1
        let mut query = vec![0.0f32; 384];
        query[0] = 0.99;
        query[1] = 0.05;

        let results = db.query_top_k(&query, 5).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk_index, 0);
        assert_eq!(results[0].content, "Rust systems programming");
        assert!(results[0].score > results[1].score);

        // Test hybrid search
        let hybrid_results = db.query_hybrid("sourdough", &query, 5, 10).unwrap();
        assert!(!hybrid_results.is_empty());
    }
}
