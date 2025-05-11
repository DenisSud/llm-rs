use backend_api::Backend;
use core::{DType, Tensor};
use cpu_backend::CpuBackend;

/// Helper: build an (n×n) identity matrix
fn identity(n: usize) -> Tensor {
    let mut t = Tensor::new(vec![n, n], DType::F32);
    let data = t.as_f32_mut();
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    t
}

/// Helper: build a 2×2 test matrix A = [[1,2],[3,4]]
fn test_matrices() -> (Tensor, Tensor) {
    let mut a = Tensor::new(vec![2, 2], DType::F32);
    let mut b = Tensor::new(vec![2, 2], DType::F32);
    a.as_f32_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_f32_mut().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    (a, b)
}

#[test]
fn matmul_identity_identity() {
    let cpu = CpuBackend::new();
    let i4 = identity(4);
    let out = cpu.matmul(&i4, &i4).expect("matmul");
    assert_eq!(out.shape, vec![4, 4]);
    assert_eq!(out.as_f32(), i4.as_f32());
}

#[test]
fn matmul_known_2x2() {
    let cpu = CpuBackend::new();
    let (a, b) = test_matrices();
    // A × B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    let expected = [19.0, 22.0, 43.0, 50.0];
    let c = cpu.matmul(&a, &b).expect("matmul 2x2");
    for (out, &exp) in c.as_f32().iter().zip(&expected) {
        assert!((out - exp).abs() < 1e-6);
    }
}

#[test]
fn softmax_row() {
    let cpu = CpuBackend::new();
    // 1×3 tensor [[0.0, 1.0, 2.0]]
    let mut t = Tensor::new(vec![1, 3], DType::F32);
    t.as_f32_mut().copy_from_slice(&[0.0, 1.0, 2.0]);
    let out = cpu.softmax(&t).expect("softmax");
    let vals = out.as_f32();
    // Manually compute softmax
    let exps: Vec<f32> = vals.iter().map(|&x| x.exp()).collect();
    let sum: f32 = exps.iter().sum();
    for &v in vals {
        let expected = v.exp() / sum;
        assert!((v - expected).abs() < 1e-6);
    }
}
