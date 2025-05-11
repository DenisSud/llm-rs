use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};
use thiserror::Error;

pub const MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub entries: u64,
    /// Offset where entries index begins
    pub index_offset: u64,
}

#[derive(Debug, Error)]
pub enum HeaderError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid magic: expected GGUF")]
    BadMagic,
}

impl GgufHeader {
    /// Parse the header from the start of the buffer
    pub fn parse(buf: &[u8]) -> Result<Self, HeaderError> {
        let mut cursor = Cursor::new(buf);

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(HeaderError::BadMagic);
        }

        // Version (u32 LE)
        let version = cursor.read_u32::<LittleEndian>()?;

        // Number of entries (u64 LE)
        let entries = cursor.read_u64::<LittleEndian>()?;

        // Current offset is start of index
        let index_offset = cursor.position();

        Ok(GgufHeader {
            version,
            entries,
            index_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_header() {
        // Build a valid header:
        // 4 bytes magic “GGUF”
        // u32 version = 1
        // u64 entries = 42
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend(&1u32.to_le_bytes());
        buf.extend(&42u64.to_le_bytes());
        // parse
        let hdr = GgufHeader::parse(&buf).expect("header parse");
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.entries, 42);
        // index_offset should point right after magic+version+entries
        assert_eq!(hdr.index_offset, 4 + 4 + 8);
    }

    #[test]
    fn parse_bad_magic() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BAD!");
        buf.extend(&0u32.to_le_bytes());
        buf.extend(&0u64.to_le_bytes());
        let err = GgufHeader::parse(&buf).unwrap_err();
        match err {
            HeaderError::BadMagic => {}
            other => panic!("expected BadMagic, got {:?}", other),
        }
    }
}
