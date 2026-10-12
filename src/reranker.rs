use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
use crate::db::SearchResult;
use crate::error::HoldonError;

pub struct Reranker {
    model: TextRerank,
}

impl Reranker {
    /// Initialize the BGE Cross-Encoder Reranker model
    pub fn try_new() -> Result<Self, HoldonError> {
        let options = RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_show_download_progress(true);

        let model = TextRerank::try_new(options)
            .map_err(|e| HoldonError::Embedding(format!("Failed to load reranker model: {e}")))?;

        Ok(Self { model })
    }

    /// Rerank retrieved candidate chunks using the cross-encoder.
    ///
    /// Returns the re-ordered results with calibrated relevance scores.
    pub fn rerank(
        &mut self,
        query: &str,
        candidates: Vec<SearchResult>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, HoldonError> {
        if candidates.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        let doc_texts: Vec<&str> = candidates.iter().map(|c| c.content.as_str()).collect();

        let rerank_results = self
            .model
            .rerank(query, &doc_texts, false, None)
            .map_err(|e| HoldonError::Embedding(format!("Failed during cross-encoder rerank: {e}")))?;

        let mut reranked = Vec::with_capacity(top_k.min(candidates.len()));
        for res in rerank_results.into_iter().take(top_k) {
            if let Some(candidate) = candidates.get(res.index) {
                let mut re_scored = candidate.clone();
                // Apply sigmoid to map raw logits into a clean [0, 1] probability range
                let sigmoid_score = 1.0 / (1.0 + (-res.score as f64).exp());
                re_scored.score = sigmoid_score;
                reranked.push(re_scored);
            }
        }

        Ok(reranked)
    }
}
