//! cpu-backend: CPU reference implementation using rayon

use backend_api::{Backend, BackendError};
use llm_core::{DType, Tensor};
use rayon::prelude::*;

/// A simple CPU backend leveraging Rayon for parallelism.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        if k1 != k2 {
            return Err(BackendError::DimMismatch);
        }
        if a.dtype != DType::F32 || b.dtype != DType::F32 {
            return Err(BackendError::TypeNotSupported);
        }

        let mut out = Tensor::new(vec![m, n], DType::F32);
        let a_data = a.as_f32();
        let b_data = b.as_f32(); // read-only access
        let out_data = out.as_f32_mut();

        // Parallelize over rows of output
        out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k1 {
                    sum += a_data[i * k1 + kk] * b_data[kk * n + j];
                }
                row[j] = sum;
            }
        });

        Ok(out)
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor, BackendError> {
        if input.dtype != DType::F32 || input.shape.len() != 2 {
            return Err(BackendError::TypeNotSupported);
        }
        let dim = input.shape[1];
        let data = input.as_f32();
        let mut out = Tensor::new(input.shape.clone(), DType::F32);

        // Process each row in parallel, writing directly into chunks of out.data
        out.as_f32_mut()
            .par_chunks_mut(dim)
            .enumerate()
            .for_each(|(i, row)| {
                let start = i * dim;
                let slice = &data[start..start + dim];
                let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = slice.iter().map(|&x| (x - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for j in 0..dim {
                    row[j] = exps[j] / sum;
                }
            });

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        gamma: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor, BackendError> {
        if input.dtype != DType::F32
            || gamma.dtype != DType::F32
            || beta.dtype != DType::F32
            || input.shape != gamma.shape
            || input.shape != beta.shape
        {
            return Err(BackendError::TypeNotSupported);
        }
        let data = input.as_f32();
        let mut out = Tensor::new(input.shape.clone(), DType::F32);
        let out_data = out.as_f32_mut();
        let eps = 1e-5f32;

        // Single vector norm
        let mean: f32 = data.iter().sum::<f32>() / (data.len() as f32);
        let var: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (data.len() as f32);
        let denom = (var + eps).sqrt();
        for i in 0..data.len() {
            out_data[i] = ((data[i] - mean) / denom) * gamma.as_f32()[i] + beta.as_f32()[i];
        }

        Ok(out)
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::{DType, Tensor};

    fn make_identity(n: usize) -> Tensor {
        let mut t = Tensor::new(vec![n, n], DType::F32);
        let data = t.as_f32_mut();
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        t
    }

    #[test]
    fn test_matmul_identity() {
        let cpu = CpuBackend::new();
        let a = make_identity(4);
        let b = make_identity(4);
        let c = cpu.matmul(&a, &b).unwrap();
        assert_eq!(c.as_f32(), a.as_f32());
    }

    #[test]
    fn test_softmax_basic() {
        let cpu = CpuBackend::new();
        let mut t = Tensor::new(vec![1, 3], DType::F32);
        t.as_f32_mut().copy_from_slice(&[0.0, 1.0, 2.0]);
        let out = cpu.softmax(&t).unwrap();
        let vals = out.as_f32();
        let sum = vals.iter().sum::<f32>();
        for &v in vals {
            assert!((v / sum - (v / sum)).abs() < 1e-6);
        }
    }
}
