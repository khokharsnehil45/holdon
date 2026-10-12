use std::fs;
use std::path::Path;
use anyhow::{Context, Result};

/// Reads file contents to text.
/// If the file ends with `.pdf` (case-insensitive), extracts text using `pdf-extract`.
/// Otherwise, reads as UTF-8 plain text.
pub fn read_document_text(path: &Path) -> Result<String> {
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if extension == "pdf" {
        let text = pdf_extract::extract_text(path).with_context(|| {
            format!(
                "Failed to extract text from PDF document '{}'. Make sure the PDF contains parseable text.",
                path.display()
            )
        })?;
        Ok(text)
    } else {
        let text = fs::read_to_string(path).with_context(|| {
            format!(
                "Failed to read file '{}'. Ensure it is valid UTF-8 encoded text.",
                path.display()
            )
        })?;
        Ok(text)
    }
}
