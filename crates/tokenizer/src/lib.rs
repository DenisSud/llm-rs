// >>> crates/tokenizer/src/lib.rs <<<
//! tokenizer: Text-to-token and token-to-text conversions

use std::collections::HashMap;

/// Simple whitespace tokenizer as placeholder
pub struct Tokenizer {
    pub vocab: HashMap<String, u32>,
    pub inv_vocab: Vec<String>,
}

impl Tokenizer {
    /// Load vocab from a file with one token per line
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut vocab = HashMap::new();
        let mut inv_vocab = Vec::new();
        for (idx, line) in content.lines().enumerate() {
            vocab.insert(line.to_string(), idx as u32);
            inv_vocab.push(line.to_string());
        }
        Ok(Tokenizer { vocab, inv_vocab })
    }

    /// Tokenize input by whitespace
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|tok| *self.vocab.get(tok).unwrap_or(&0))
            .collect()
    }

    /// Detokenize token IDs into text
    pub fn detokenize(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| {
                self.inv_vocab
                    .get(i as usize)
                    .cloned()
                    .unwrap_or_else(|| "<unk>".to_string())
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_whitespace_tokenizer_roundtrip() -> anyhow::Result<()> {
        // Prepare a temporary vocab file
        let mut f = NamedTempFile::new()?;
        writeln!(f, "hello")?;
        writeln!(f, "world")?;
        writeln!(f, "foo")?;
        // Build tokenizer
        let tok = Tokenizer::from_file(f.path())?;
        // Test tokenize
        let ids = tok.tokenize("hello foo unknown");
        // "hello"->0, "foo"->2, "unknown"->fallback 0
        assert_eq!(ids, vec![0, 2, 0]);
        // Test detokenize
        let txt = tok.detokenize(&[1, 0, 2]);
        assert_eq!(txt, "world hello foo");
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_from_file_not_found() {
        // should panic (via anyhow) if path doesn't exist
        let _ = Tokenizer::from_file(std::path::Path::new("/no/such/file.txt")).unwrap();
    }
}
