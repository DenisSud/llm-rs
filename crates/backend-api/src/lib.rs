// >>> crates/backend-api/src/lib.rs <<<
//! backend-api: Defines Backend trait for tensor operations

use llm_core::DType;
use llm_core::Tensor;
use thiserror::Error;

/// Errors returned by compute backends
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("dimension mismatch")]
    DimMismatch,
    #[error("data type not supported")]
    TypeNotSupported,
    #[error("device unavailable")]
    DeviceUnavailable,
    #[error("not implemented")]
    NotImplemented,
}

/// GPU or CPU backend abstraction
pub trait Backend {
    /// Matrix multiplication: A (m×k) × B (k×n) = out (m×n)
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError>;

    /// Softmax over last dimension of a rank-2 tensor
    fn softmax(&self, input: &Tensor) -> Result<Tensor, BackendError>;

    /// Layer normalization
    fn layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyBackend;

    impl Backend for DummyBackend {
        fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, BackendError> {
            Err(BackendError::NotImplemented)
        }
        fn softmax(&self, _input: &Tensor) -> Result<Tensor, BackendError> {
            Err(BackendError::NotImplemented)
        }
        fn layer_norm(
            &self,
            _input: &Tensor,
            _g: &Tensor,
            _b: &Tensor,
        ) -> Result<Tensor, BackendError> {
            Err(BackendError::NotImplemented)
        }
    }

    #[test]
    fn dummy_backend_methods() {
        // We just ensure each returns NotImplemented
        let d = DummyBackend;
        // Make empty tensors to satisfy signature
        let empty = Tensor::new(vec![0], DType::F32);
        assert!(matches!(
            d.matmul(&empty, &empty),
            Err(BackendError::NotImplemented)
        ));
        assert!(matches!(
            d.softmax(&empty),
            Err(BackendError::NotImplemented)
        ));
        assert!(matches!(
            d.layer_norm(&empty, &empty, &empty),
            Err(BackendError::NotImplemented)
        ));
    }
}
