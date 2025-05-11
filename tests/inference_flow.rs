use inference::Model;
use std::path::Path;

#[test]
fn load_and_run_dummy_model() {
    // You’ll need a tiny GGUF test model (e.g., 2×2 identity weights).
    let model = Model::load(Path::new("tests/data/identity.gguf")).unwrap();
    let output = model.forward("Hello").unwrap();
    // With identity weights and simplistic tokenizer, output == input tokens
    assert_eq!(output, vec![/* expected token IDs for "Hello" */]);
}
