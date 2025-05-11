use std::path::Path;

/// Data types supported for tensor elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    // BF16, // Will add later if needed for broader compatibility
    // I8,   // Keeping quantization out for now
    // U8,
}

/// Represents the underlying data storage for a Tensor.
/// For now, we'll use an owned Vec<u8> and assume the user or system
/// ensures correct byte interpretation according to DType.
#[derive(Debug, Clone)]
pub enum TensorData {
    Owned(Vec<u8>),
    // Mapped(MmapData<'a>), // For future mmap support
}

/// A multi-dimensional array (tensor).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: TensorData,
    pub name: Option<String>, // Optional name for debugging or linking
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: DType, data: Vec<u8>, name: Option<String>) -> Self {
        Tensor {
            shape,
            dtype,
            data: TensorData::Owned(data),
            name,
        }
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_in_bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 => 2,
        };
        self.num_elements() * bytes_per_element
    }

    // Basic accessor for f32 data, assuming DType is F32 and correct alignment.
    // More robust accessors would handle errors and other DTypes.
    // This is a simplified example.
    pub fn as_f32_slice(&self) -> Result<&[f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DataTypeMismatch);
        }
        match &self.data {
            TensorData::Owned(bytes) => {
                // Ensure the byte slice can be safely transmuted.
                // This requires alignment and size to be correct.
                if bytes.len() % 4 != 0 || bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
                    return Err(TensorError::AlignmentError);
                }
                // SAFETY: We've checked DType, alignment, and size (implicitly by num_elements * item_size).
                // This is a common pattern but requires care.
                Ok(unsafe { 
                    std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
                })
            }
        }
    }

    // Placeholder for a method to get mutable f32 slice
    pub fn as_f32_slice_mut(&mut self) -> Result<&mut [f32], TensorError> {
        if self.dtype != DType::F32 {
            return Err(TensorError::DataTypeMismatch);
        }
        match &mut self.data {
            TensorData::Owned(bytes) => {
                if bytes.len() % 4 != 0 || bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
                    return Err(TensorError::AlignmentError);
                }
                Ok(unsafe { 
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, bytes.len() / 4)
                })
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Data type mismatch for tensor operation")]
    DataTypeMismatch,
    #[error("Tensor data is not properly aligned for the requested type")]
    AlignmentError,
    #[error("Shape mismatch for tensor operation: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    #[error("Invalid dimension index: {index}, for tensor with {dims} dimensions")]
    InvalidDimension { index: usize, dims: usize },
}

// General error type for the core library, can be expanded.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("Tensor error: {0}")]
    Tensor(#[from] TensorError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

// Basic logging utilities could be added here or in a separate `utils` crate.
// For now, we'll rely on direct use of `log` crate facilities if needed in other crates.

// If we add a utils crate later, we might move this or expand it.
pub fn ensure_parent_dir_exists(file_path: &Path) -> std::io::Result<()> {
    if let Some(parent_dir) = file_path.parent() {
        if !parent_dir.exists() {
            std::fs::create_dir_all(parent_dir)?;
        }
    }
    Ok(())
}

// Example of a utility function that might live in core or utils
pub fn calculate_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
} 