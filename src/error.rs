use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HoldonError {
    #[error("I/O error at '{path}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Embedding engine error: {0}")]
    Embedding(String),

    #[error("Configuration / path resolution error: {0}")]
    Config(String),

    #[error("Invalid parameter: {0}")]
    InvalidInput(String),
}
