use byteorder::ReadBytesExt;
use memmap2::Mmap;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};
use thiserror::Error;

use crate::{
    header::{GgufHeader, HeaderError},
    tensor::{GgufTensor, TensorError},
};

/// The loaded model: metadata strings + weight tensors
pub struct GgufModel {
    pub metadata: HashMap<String, String>,
    pub tensors: HashMap<String, llm_core::Tensor>,
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("header parse error: {0}")]
    Header(#[from] HeaderError),
    #[error("tensor parse error: {0}")]
    Tensor(#[from] TensorError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl GgufModel {
    /// Load a .gguf file, parse header, metadata, and tensors
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GgufError> {
        // Memory-map the entire file
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let buf = &mmap[..];

        // Parse header
        let header = GgufHeader::parse(buf)?;

        // Seek into the index
        let mut cursor = std::io::Cursor::new(buf);
        cursor.seek(SeekFrom::Start(header.index_offset))?;

        // Prepare outputs
        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();

        // Iterate entries
        for _ in 0..header.entries {
            let kind = cursor.read_u32::<byteorder::LittleEndian>()?;
            match kind {
                0 => {
                    // metadata: key, value
                    let klen = cursor.read_u32::<byteorder::LittleEndian>()? as usize;
                    let mut kbuf = vec![0u8; klen];
                    cursor.read_exact(&mut kbuf)?;
                    let key = String::from_utf8_lossy(&kbuf).into_owned();

                    let vlen = cursor.read_u32::<byteorder::LittleEndian>()? as usize;
                    let mut vbuf = vec![0u8; vlen];
                    cursor.read_exact(&mut vbuf)?;
                    let val = String::from_utf8_lossy(&vbuf).into_owned();

                    metadata.insert(key, val);
                }
                1 => {
                    // tensor
                    let gt = GgufTensor::parse_from(&mut cursor)?;
                    let tensor = gt.load(buf)?;
                    tensors.insert(gt.name.clone(), tensor);
                }
                other => {
                    return Err(GgufError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("unknown entry kind {}", other),
                    )));
                }
            }
        }

        Ok(GgufModel { metadata, tensors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build and write a minimal GGUF with:
    /// - header: GGUF, version=1, entries=2
    /// - entry0: metadata k->v
    /// - entry1: one U8 tensor "t" of length 3, data [10,20,30]
    fn write_minimal_gguf(path: &Path) {
        let mut file = File::create(path).unwrap();
        let mut buf = Vec::new();

        // header
        buf.extend_from_slice(b"GGUF");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u64::<LittleEndian>(2).unwrap();

        // entry0 metadata
        buf.write_u32::<LittleEndian>(0).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.extend_from_slice(b"k");
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.extend_from_slice(b"v");

        // entry1 tensor
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.extend_from_slice(b"t");
        buf.write_u32::<LittleEndian>(1).unwrap(); // ndims
        buf.write_u64::<LittleEndian>(3).unwrap(); // dim[0]
        buf.write_u32::<LittleEndian>(4).unwrap(); // code=4 => U8

        // compute offset relative to file start
        let idx_end = buf.len() as u64;
        // we next write offset and size
        let data_offset = idx_end + 16; // skip offset(8)+size(8)
        buf.write_u64::<LittleEndian>(data_offset).unwrap();
        buf.write_u64::<LittleEndian>(3).unwrap();

        file.write_all(&buf).unwrap();
        file.write_all(&[10u8, 20, 30]).unwrap();
    }

    #[test]
    fn load_minimal_model() {
        let tmp = NamedTempFile::new().unwrap();
        write_minimal_gguf(tmp.path());
        let model = GgufModel::load(tmp.path()).expect("load");
        assert_eq!(model.metadata.get("k").map(|s| s.as_str()), Some("v"));
        let t = model.tensors.get("t").unwrap();
        assert_eq!(t.shape, vec![3]);
        assert_eq!(t.dtype, llm_core::DType::U8);
        assert_eq!(&t.data[..], &[10, 20, 30]);
    }
}
