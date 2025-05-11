use byteorder::{LittleEndian, ReadBytesExt};
use llm_core::{DType, Tensor};
use std::io::{Read, Seek};
use thiserror::Error;

/// Represents one entry in the GGUF index
#[derive(Debug)]
pub enum GgufEntry {
    Metadata { key: String, value: String },
    Tensor(GgufTensor),
}

/// Describes a tensor stored in GGUF
#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data_offset: u64,
    pub data_size: u64,
}

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported dtype code: {0}")]
    UnsupportedDType(u32),
}

impl GgufTensor {
    /// Parse one tensor entry from the index cursor (after reading the kind=1 tag)
    pub fn parse_from<R: Read + Seek>(cursor: &mut R) -> Result<Self, TensorError> {
        // name
        let name_len = cursor.read_u32::<LittleEndian>()? as usize;
        let mut name_buf = vec![0u8; name_len];
        cursor.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).into_owned();

        // shape
        let ndims = cursor.read_u32::<LittleEndian>()?;
        let mut shape = Vec::with_capacity(ndims as usize);
        for _ in 0..ndims {
            shape.push(cursor.read_u64::<LittleEndian>()? as usize);
        }

        // dtype
        let code = cursor.read_u32::<LittleEndian>()?;
        let dtype = match code {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::BF16,
            3 => DType::I8,
            4 => DType::U8,
            other => return Err(TensorError::UnsupportedDType(other)),
        };

        // number of bytes per element (not used directly but can sanity-check)
        let _elem_bytes = match dtype {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
        };

        // data location
        let data_offset = cursor.read_u64::<LittleEndian>()?;
        let data_size = cursor.read_u64::<LittleEndian>()?;

        Ok(GgufTensor {
            name,
            shape,
            dtype,
            data_offset,
            data_size,
        })
    }

    /// Load the tensor’s raw bytes into an `llm_core::Tensor`
    pub fn load(&self, mmap: &[u8]) -> Result<Tensor, TensorError> {
        let start = self.data_offset as usize;
        let end = start + (self.data_size as usize);
        let slice = &mmap[start..end];

        // Create host tensor and copy bytes
        let mut tensor = Tensor::new(self.shape.clone(), self.dtype);
        tensor.data.copy_from_slice(slice);
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Cursor;

    fn make_index_entry() -> Vec<u8> {
        // We build only the bytes *after* kind=1:
        // name_len=3, name="foo"
        // ndims=2, dims=[2,3]
        // code=0 (F32), offset=100, size=2*3*4=24
        let mut v = Vec::new();
        v.write_u32::<LittleEndian>(3).unwrap();
        v.extend_from_slice(b"foo");
        v.write_u32::<LittleEndian>(2).unwrap();
        v.write_u64::<LittleEndian>(2).unwrap();
        v.write_u64::<LittleEndian>(3).unwrap();
        v.write_u32::<LittleEndian>(0).unwrap();
        v.write_u64::<LittleEndian>(100).unwrap();
        v.write_u64::<LittleEndian>(24).unwrap();
        v
    }

    #[test]
    fn parse_tensor_entry() {
        let bytes = make_index_entry();
        let mut cursor = Cursor::new(&bytes[..]);
        let gg = GgufTensor::parse_from(&mut cursor).unwrap();
        assert_eq!(gg.name, "foo");
        assert_eq!(gg.shape, vec![2, 3]);
        assert_eq!(gg.dtype, DType::F32);
        assert_eq!(gg.data_offset, 100);
        assert_eq!(gg.data_size, 24);
    }
}
