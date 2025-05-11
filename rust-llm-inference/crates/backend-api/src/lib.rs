use core::Tensor;

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("Operation not supported by this backend: {0}")]
    UnsupportedOperation(String),
    #[error("Backend execution error: {0}")]
    ExecutionError(String),
    #[error("Failed to setup backend: {0}")]
    SetupError(String),
    #[error("Tensor error: {0}")]
    TensorError(#[from] core::TensorError),
    #[error("Core error: {0}")]
    CoreError(#[from] core::CoreError),
    #[error("Incompatible tensor for backend operation: {0}")]
    IncompatibleTensor(String),
}

/// Trait defining the operations a compute backend must support.
pub trait Backend: Send + Sync {
    /// Matrix multiplication: C = A * B
    fn matmul(&self, a: &Tensor, b: &Tensor, b_transposed: bool) -> Result<Tensor, BackendError>;

    /// Softmax, typically applied to the last dimension of the input tensor.
    fn softmax(&self, input: &Tensor, axis: usize) -> Result<Tensor, BackendError>;

    /// Layer normalization.
    fn layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor, // scale
        beta: &Tensor,  // bias
        epsilon: f32,
    ) -> Result<Tensor, BackendError>;
    
    /// RMS normalization.
    fn rms_norm(
        &self, 
        input: &Tensor, 
        weights: &Tensor, // scale
        epsilon: f32
    ) -> Result<Tensor, BackendError>;

    /// Element-wise addition: C = A + B (supports broadcasting)
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise multiplication: C = A * B (supports broadcasting)
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError>;

    /// Element-wise division: C = A / B (supports broadcasting)
    fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError>;

    /// SiLU activation function.
    fn silu(&self, input: &Tensor) -> Result<Tensor, BackendError>;

    /// Performs embedding lookup.
    fn embedding(&self, ids: &[u32], embedding_matrix: &Tensor) -> Result<Tensor, BackendError>;

    // TODO: Add RoPE (Rotary Position Embedding) if it's a standalone op.
    // Otherwise, it might be fused or handled within attention logic.

    // TODO: Consider if ops like `cat`, `transpose`, `reshape` are needed at the backend level
    // or if they can be handled by manipulating `core::Tensor` metadata and data directly
    // (potentially creating new Tensors that view or copy data).

    /// Returns a descriptive name for the backend (e.g., "cpu", "wgpu").
    fn name(&self) -> String;
}

/// Enum to allow choosing a backend at runtime.
pub enum BackendChoice {
    Cpu,
    Wgpu,
    // Auto, // Could try to auto-detect best available
} 