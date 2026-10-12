use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use crate::error::HoldonError;

pub const EMBEDDING_DIM: usize = 384;

pub struct Embedder {
    model: TextEmbedding,
}

impl Embedder {
    /// Initialize the AllMiniLML6V2 local embedding model.
    pub fn try_new() -> Result<Self, HoldonError> {
        let options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true);

        let model = TextEmbedding::try_new(options)
            .map_err(|e| HoldonError::Embedding(format!("Failed to load embedding model: {e}")))?;

        Ok(Self { model })
    }

    /// Batch embeds a slice of text strings into normalized f32 vectors.
    pub fn embed_batch<S: AsRef<str>>(&mut self, texts: &[S]) -> Result<Vec<Vec<f32>>, HoldonError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let str_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let embeddings = self.model
            .embed(str_refs, None)
            .map_err(|e| HoldonError::Embedding(format!("Failed to compute embeddings: {e}")))?;

        Ok(embeddings)
    }

    /// Embeds a single query string.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, HoldonError> {
        let mut results = self.embed_batch(&[query])?;
        results
            .pop()
            .ok_or_else(|| HoldonError::Embedding("Model returned empty embedding vector".to_string()))
    }
}
