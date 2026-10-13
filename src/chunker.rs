use rayon::prelude::*;
use crate::embedder::Embedder;
use crate::error::HoldonError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    pub content: String,
    pub chunk_index: usize,
}

/// Splits input text into sliding windows of characters with overlap (fixed-size chunking).
/// Uses Rayon parallel iterators across sliding window chunk boundaries for multi-core performance.
pub fn chunk_text_fixed(
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<TextChunk>, HoldonError> {
    if chunk_size == 0 {
        return Err(HoldonError::InvalidInput(
            "chunk_size must be greater than 0".to_string(),
        ));
    }
    if overlap >= chunk_size {
        return Err(HoldonError::InvalidInput(format!(
            "overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
        )));
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return Ok(Vec::new());
    }

    let step = chunk_size - overlap;
    let total_chars = chars.len();

    // Compute chunk window start indices
    let mut starts = Vec::new();
    let mut start = 0;
    while start < total_chars {
        starts.push(start);
        let end = (start + chunk_size).min(total_chars);
        if end == total_chars {
            break;
        }
        start += step;
    }

    // Extract chunk strings in parallel across available CPU cores
    let raw_chunks: Vec<String> = starts
        .into_par_iter()
        .map(|s| {
            let end = (s + chunk_size).min(total_chars);
            chars[s..end].iter().collect::<String>()
        })
        .collect();

    // Filter non-empty and assign sequential indices
    let mut chunks = Vec::with_capacity(raw_chunks.len());
    let mut chunk_index = 0;
    for content in raw_chunks {
        if !content.trim().is_empty() {
            chunks.push(TextChunk {
                content,
                chunk_index,
            });
            chunk_index += 1;
        }
    }

    Ok(chunks)
}

/// Computes cosine similarity between two float vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Breaks text into sentences or natural segments based on punctuation (. ! ? \n\n).
/// Splits text into natural paragraph boundaries in parallel, then gathers sentences.
pub fn split_sentences(text: &str) -> Vec<String> {
    // Split into paragraphs / major blocks
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    // Process sentence parsing across paragraphs in parallel
    let nested: Vec<Vec<String>> = paragraphs
        .into_par_iter()
        .map(|para| {
            let mut sents = Vec::new();
            let mut current = String::new();
            let chars: Vec<char> = para.chars().collect();
            let len = chars.len();
            let mut i = 0;

            while i < len {
                let c = chars[i];
                current.push(c);

                let is_term = (c == '.' || c == '!' || c == '?' || c == '\n')
                    && (i + 1 == len || chars[i + 1].is_whitespace());

                if is_term {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        sents.push(trimmed.to_string());
                    }
                    current.clear();
                }
                i += 1;
            }

            let trimmed = current.trim();
            if !trimmed.is_empty() {
                sents.push(trimmed.to_string());
            }

            sents
        })
        .collect();

    nested.into_iter().flatten().collect()
}

/// Semantic chunker:
/// 1. Splits text into coherent sentences in parallel.
/// 2. Batch embeds all sentences using ONNX.
/// 3. Computes consecutive similarities in parallel.
/// 4. Groups sentences into semantic units when meaning shifts below threshold.
pub fn chunk_text_semantic(
    text: &str,
    embedder: &mut Embedder,
    similarity_threshold: f32,
    max_chunk_chars: usize,
    min_chunk_chars: usize,
) -> Result<Vec<TextChunk>, HoldonError> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    if sentences.len() == 1 {
        return Ok(vec![TextChunk {
            content: sentences[0].clone(),
            chunk_index: 0,
        }]);
    }

    // Embed all individual sentences in a single optimized batch
    let embeddings = embedder.embed_batch(&sentences)?;

    // Compute pairwise cosine similarities in parallel across threads
    let sims: Vec<f32> = (0..sentences.len() - 1)
        .into_par_iter()
        .map(|i| cosine_similarity(&embeddings[i], &embeddings[i + 1]))
        .collect();

    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_chars = 0;
    let mut chunk_index = 0;

    for i in 0..sentences.len() {
        let sent = &sentences[i];
        let sent_len = sent.chars().count();

        if current_chunk.is_empty() {
            current_chunk.push(sent.as_str());
            current_chars = sent_len;
            continue;
        }

        let sim = sims[i - 1];
        let would_exceed_max = (current_chars + 1 + sent_len) > max_chunk_chars;
        let meets_min = current_chars >= min_chunk_chars;

        let should_split = (meets_min && sim < similarity_threshold) || would_exceed_max;

        if should_split {
            let combined = current_chunk.join(" ");
            chunks.push(TextChunk {
                content: combined,
                chunk_index,
            });
            chunk_index += 1;
            current_chunk.clear();
            current_chunk.push(sent.as_str());
            current_chars = sent_len;
        } else {
            current_chunk.push(sent.as_str());
            current_chars += 1 + sent_len;
        }
    }

    if !current_chunk.is_empty() {
        let combined = current_chunk.join(" ");
        chunks.push(TextChunk {
            content: combined,
            chunk_index,
        });
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunking_fixed() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk_text_fixed(text, 10, 2).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].content, "abcdefghij");
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].content, "ijklmnopqr");
        assert_eq!(chunks[2].content, "qrstuvwxyz");
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. How are you? Rust is fast!\n\nNew paragraph.";
        let sents = split_sentences(text);
        assert_eq!(sents.len(), 4);
        assert_eq!(sents[0], "Hello world.");
        assert_eq!(sents[1], "How are you?");
        assert_eq!(sents[2], "Rust is fast!");
        assert_eq!(sents[3], "New paragraph.");
    }
}
