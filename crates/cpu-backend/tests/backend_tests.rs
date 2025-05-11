use backend_api::Backend;
use cpu_backend::CpuBackend;
use llm_core::{DType, Tensor};

/// Same 2×2 helper
fn test_matrices() -> (Tensor, Tensor) {
    let mut a = Tensor::new(vec![2, 2], DType::F32);
    let mut b = Tensor::new(vec![2, 2], DType::F32);
    a.as_f32_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
    b.as_f32_mut().copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
    (a, b)
}

#[test]
fn matmul_wgpu_2x2() {
    let (a, b) = test_matrices();
    let cpu = CpuBackend::new();
    let c = cpu.matmul(&a, &b).expect("matmul");

    // Expected [[19,22],[43,50]]
    let expected = [19.0, 22.0, 43.0, 50.0];
    for (&out, &exp) in c.as_f32().iter().zip(&expected) {
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

    // Compute expected softmax of [0,1,2]
    let input = &[0.0f32, 1.0, 2.0];
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let expected: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Compare each output element
    for i in 0..3 {
        assert!(
            (vals[i] - expected[i]).abs() < 1e-6,
            "softmax[{}] = {}, but expected {}",
            i,
            vals[i],
            expected[i]
        );
    }
}
