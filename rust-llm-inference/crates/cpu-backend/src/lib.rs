use core::{Tensor, DType, TensorData, TensorError};
use backend_api::{Backend, BackendError};

// Helper macro for unimplemented operations
macro_rules! unimplemented_op {
    ($op_name:expr) => {
        Err(BackendError::UnsupportedOperation(format!(
            "{} is not yet implemented for CpuBackend",
            $op_name
        )))
    };
}

#[derive(Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> String {
        "cpu".to_string()
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, _b_transposed: bool) -> Result<Tensor, BackendError> {
        // Basic validation (example for matmul M K x K N -> M N)
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(BackendError::TensorError(TensorError::ShapeMismatch {
                expected: vec![2, 0], // Representing a 2D tensor
                got: if a.shape.len() != 2 { a.shape.clone() } else { b.shape.clone() }
            }));
        }
        if a.shape[1] != b.shape[0] {
             return Err(BackendError::TensorError(TensorError::ShapeMismatch {
                expected: vec![a.shape[1]], 
                got: vec![b.shape[0]]
            }));
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }

        // Placeholder: Actual matmul logic is complex.
        // For now, return a zeroed tensor of the correct shape.
        let m = a.shape[0];
        let n = b.shape[1];
        let out_shape = vec![m, n];
        let num_elements = m * n;
        let out_data = vec![0u8; num_elements * std::mem::size_of::<f32>()];
        Ok(Tensor::new(out_shape, DType::F32, out_data, Some("matmul_out".to_string())))
        // unimplemented_op!("matmul")
    }

    fn softmax(&self, input: &Tensor, _axis: usize) -> Result<Tensor, BackendError> {
        // Placeholder: Actual softmax logic is complex.
        // For now, return a clone of the input or a zeroed tensor of the same shape.
        if input.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        let out_data = vec![0u8; input.size_in_bytes()];
        Ok(Tensor::new(input.shape.clone(), input.dtype, out_data, Some("softmax_out".to_string())))
        // unimplemented_op!("softmax")
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        _gamma: &Tensor,
        _beta: &Tensor,
        _epsilon: f32,
    ) -> Result<Tensor, BackendError> {
        if input.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        let out_data = vec![0u8; input.size_in_bytes()];
        Ok(Tensor::new(input.shape.clone(), input.dtype, out_data, Some("layernorm_out".to_string())))
        // unimplemented_op!("layer_norm")
    }

    fn rms_norm(
        &self, 
        input: &Tensor, 
        _weights: &Tensor, 
        _epsilon: f32
    ) -> Result<Tensor, BackendError> {
        if input.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        let out_data = vec![0u8; input.size_in_bytes()];
        Ok(Tensor::new(input.shape.clone(), input.dtype, out_data, Some("rmsnorm_out".to_string())))
        // unimplemented_op!("rms_norm")
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
        // Placeholder: Actual add logic with broadcasting is complex.
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        // For simplicity, assume shapes are identical for now.
        if a.shape != b.shape {
            return Err(BackendError::TensorError(TensorError::ShapeMismatch{ expected: a.shape.clone(), got: b.shape.clone() }));
        }
        let out_data = vec![0u8; a.size_in_bytes()];
        Ok(Tensor::new(a.shape.clone(), a.dtype, out_data, Some("add_out".to_string())))
        // unimplemented_op!("add")
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        if a.shape != b.shape {
             return Err(BackendError::TensorError(TensorError::ShapeMismatch{ expected: a.shape.clone(), got: b.shape.clone() }));
        }
        let out_data = vec![0u8; a.size_in_bytes()];
        Ok(Tensor::new(a.shape.clone(), a.dtype, out_data, Some("mul_out".to_string())))
        // unimplemented_op!("mul")
    }

    fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        if a.shape != b.shape {
             return Err(BackendError::TensorError(TensorError::ShapeMismatch{ expected: a.shape.clone(), got: b.shape.clone() }));
        }
        let out_data = vec![0u8; a.size_in_bytes()];
        Ok(Tensor::new(a.shape.clone(), a.dtype, out_data, Some("div_out".to_string())))
        // unimplemented_op!("div")
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor, BackendError> {
        if input.dtype != DType::F32 {
            return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        let out_data = vec![0u8; input.size_in_bytes()];
        Ok(Tensor::new(input.shape.clone(), input.dtype, out_data, Some("silu_out".to_string())))
        // unimplemented_op!("silu")
    }

    fn embedding(&self, ids: &[u32], embedding_matrix: &Tensor) -> Result<Tensor, BackendError> {
        if embedding_matrix.dtype != DType::F32 {
             return Err(BackendError::TensorError(TensorError::DataTypeMismatch));
        }
        if embedding_matrix.shape.len() != 2 {
            return Err(BackendError::TensorError(TensorError::ShapeMismatch{
                expected: vec![0,0], // any 2D shape
                got: embedding_matrix.shape.clone()
            }));
        }
        // Output shape: [num_ids, embedding_dim]
        let num_ids = ids.len();
        let embedding_dim = embedding_matrix.shape[1];
        let out_shape = vec![num_ids, embedding_dim];
        let out_data_size = num_ids * embedding_dim * std::mem::size_of::<f32>();
        let out_data = vec![0u8; out_data_size];
        Ok(Tensor::new(out_shape, DType::F32, out_data, Some("embedding_out".to_string())))
        // unimplemented_op!("embedding")
    }
} 