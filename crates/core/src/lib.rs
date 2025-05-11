// >>> crates/core/src/lib.rs <<<
//! llm-core: Definitions for Tensor, shapes, and dtypes

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    U8,
}

/// Multi-dimensional tensor container
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl Tensor {
    /// Number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Create a new zeroed tensor
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        let elem_size = match dtype {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 => 1,
        };
        let bytes = shape.iter().product::<usize>() * elem_size;
        Tensor {
            shape,
            dtype,
            data: vec![0u8; bytes],
        }
    }

    /// View as f32 slice
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32);
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.num_elements()) }
    }

    /// Mutable f32 slice
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32);
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, self.num_elements())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_elements_and_bytes() {
        let t = Tensor::new(vec![4, 5, 6], DType::F32);
        assert_eq!(t.num_elements(), 4 * 5 * 6);
        // 4×5×6 elements × 4 bytes per f32
        assert_eq!(t.data.len(), 4 * 5 * 6 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_as_f32_accessors() {
        let mut t = Tensor::new(vec![2, 3], DType::F32);
        let slice = t.as_f32_mut();
        assert_eq!(slice.len(), 6);
        for i in 0..slice.len() {
            slice[i] = (i * 10) as f32;
        }
        let read = t.as_f32();
        for (i, &val) in read.iter().enumerate() {
            assert_eq!(val, (i * 10) as f32);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_as_f32_wrong_dtype() {
        let t = Tensor::new(vec![2, 2], DType::I8);
        let _ = t.as_f32(); // should panic because dtype!=F32
    }
}
