use crate::embedder::Embedder;
use crate::error::HoldonError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    pub content: String,
    pub chunk_index: usize,
}

/// Splits input text into sliding windows of characters with overlap (fixed-size chunking).
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
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chunk_index = 0;

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk_chars = &chars[start..end];
        let chunk_string: String = chunk_chars.iter().collect();

        if !chunk_string.trim().is_empty() {
            chunks.push(TextChunk {
                content: chunk_string,
                chunk_index,
            });
            chunk_index += 1;
        }

        if end == chars.len() {
            break;
        }
        start += step;
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
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        current.push(c);

        let is_term = c == '.' || c == '!' || c == '?' || c == '\n';
        if is_term {
            // Check double newline
            let is_double_newline = c == '\n' && (i + 1 < len && chars[i + 1] == '\n');
            let is_sentence_end = (c == '.' || c == '!' || c == '?')
                && (i + 1 == len || chars[i + 1].is_whitespace());

            if is_double_newline || is_sentence_end {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                current.clear();
                if is_double_newline {
                    i += 1; // skip consecutive newline
                }
            }
        }
        i += 1;
    }

    let trimmed = current.trim();
    if !trimmed.is_empty() {
        sentences.push(trimmed.to_string());
    }

    sentences
}

/// Semantic chunker:
/// 1. Splits text into coherent sentences.
/// 2. Embeds each sentence.
/// 3. Computes cosine similarity between consecutive sentences.
/// 4. When semantic similarity drops below `similarity_threshold` (or size bounds exceeded),
///    starts a new chunk.
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

    // Embed all individual sentences in a single batch
    let embeddings = embedder.embed_batch(&sentences)?;

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

        let sim = cosine_similarity(&embeddings[i - 1], &embeddings[i]);
        let would_exceed_max = (current_chars + 1 + sent_len) > max_chunk_chars;
        let meets_min = current_chars >= min_chunk_chars;

        // Split condition: semantic distance jump OR max size exceeded
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
