use std::path::Path;
use std::sync::Arc;

use core::{Tensor, DType, CoreError};
use gguf_parser::{GgufModel, GgufParserError};
use tokenizer::{Tokenizer, TokenizerError};
use backend_api::{Backend, BackendChoice, BackendError};
use cpu_backend::CpuBackend;
use wgpu_backend::WgpuBackend;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("GGUF parsing failed: {0}")]
    Gguf(#[from] GgufParserError),
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] TokenizerError),
    #[error("Backend error: {0}")]
    Backend(#[from] BackendError),
    #[error("Core system error: {0}")]
    Core(#[from] CoreError),
    #[error("Model configuration error: {0}")]
    ModelConfig(String),
    #[error("Required tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Inference logic error: {0}")]
    LogicError(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
     #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

/// High-level model API for performing inference.
pub struct Model {
    gguf_model: Arc<GgufModel>, // Arc for potential sharing if Model is cloned or sent across threads
    backend: Box<dyn Backend>,
    tokenizer: Tokenizer,
    // Model-specific configuration extracted from metadata, e.g.,
    // num_layers: usize,
    // num_heads: usize,
    // embedding_dim: usize,
    // vocab_size: usize,
}

impl Model {
    /// Loads a model from a GGUF file path and prepares it for inference with the chosen backend.
    pub fn load(model_path: &Path, tokenizer_vocab_path: &Path, backend_choice: BackendChoice) -> Result<Self, InferenceError> {
        // log::info!("Loading GGUF model from: {:?}", model_path);
        let gguf_model = Arc::new(gguf_parser::load(model_path)?);

        // log::info!("Loading tokenizer from: {:?}", tokenizer_vocab_path);
        let tokenizer = Tokenizer::load(tokenizer_vocab_path)?;

        // log::info!("Initializing backend: {:?}", backend_choice);
        let backend: Box<dyn Backend> = match backend_choice {
            BackendChoice::Cpu => Box::new(CpuBackend::new()),
            BackendChoice::Wgpu => {
                // WgpuBackend::new() might return a Result, handle it here.
                Box::new(WgpuBackend::new().map_err(InferenceError::Backend)?)
            }
        };
        // log::info!("Using backend: {}", backend.name());

        // TODO: Extract necessary model configuration from gguf_model.metadata
        // e.g., num_layers, num_heads, hidden_dim, etc.
        // For now, these are not stored in the Model struct directly.

        Ok(Model {
            gguf_model,
            backend,
            tokenizer,
        })
    }

    /// Performs a forward pass (inference) on the given prompt to predict the next token(s).
    /// For simplicity, this currently aims to return a sequence of predicted token IDs.
    /// A more complete implementation would handle sampling strategies (argmax, top-k, etc.)
    /// and could return generated text or more detailed output.
    pub fn forward(&self, prompt: &str) -> Result<Vec<u32>, InferenceError> {
        // log::info!("Starting forward pass for prompt: \"{}\"", prompt);

        // 1. Tokenize the prompt
        let token_ids = self.tokenizer.tokenize(prompt)?;
        // log::debug!("Tokenized IDs: {:?}", token_ids);
        if token_ids.is_empty() {
            return Ok(Vec::new()); // Or an error if empty prompt is not allowed
        }

        // 2. Embedding lookup for the token IDs
        //    Requires an "embedding_weight" tensor (name might vary, e.g., "tok_embeddings.weight")
        let embedding_tensor_name = "model.embed_tokens.weight"; // Example name, actual name from GGUF
        let embedding_weights = self.gguf_model.get_tensor(embedding_tensor_name)
            .ok_or_else(|| InferenceError::TensorNotFound(embedding_tensor_name.to_string()))?;
        
        let mut current_hidden_state = self.backend.embedding(&token_ids, embedding_weights)?;
        // log::debug!("Initial embedding shape: {:?}", current_hidden_state.shape);

        // --- Simplified Transformer Block Loop Placeholder ---
        // A real implementation would loop through N layers, applying attention and MLP blocks.
        // For each block, it would fetch specific weight tensors from `self.gguf_model`.
        // Example structure (highly simplified):
        let num_layers = self.gguf_model.metadata.get("llama.block_count").map_or(1, |s| s.parse().unwrap_or(1)); // Dummy read

        for i in 0..num_layers {
            // Fetch layer-specific weights using names like:
            // `model.layers.{i}.input_layernorm.weight`
            // `model.layers.{i}.self_attn.q_proj.weight`
            // `model.layers.{i}.self_attn.k_proj.weight`
            // `model.layers.{i}.self_attn.v_proj.weight`
            // `model.layers.{i}.self_attn.o_proj.weight`
            // `model.layers.{i}.post_attention_layernorm.weight`
            // `model.layers.{i}.mlp.gate_proj.weight`
            // `model.layers.{i}.mlp.up_proj.weight`
            // `model.layers.{i}.mlp.down_proj.weight`

            // Dummy RMSNorm (or LayerNorm)
            let norm_weights_name = format!("model.layers.{}.input_layernorm.weight", i);
            let norm_weights = self.gguf_model.get_tensor(&norm_weights_name)
                .ok_or_else(|| InferenceError::TensorNotFound(norm_weights_name.clone()))?;
            current_hidden_state = self.backend.rms_norm(&current_hidden_state, norm_weights, 1e-6f32)?;

            // Dummy MatMul for QKV (super simplified - would be 3 matmuls + attention logic)
            let q_proj_name = format!("model.layers.{}.self_attn.q_proj.weight", i);
            let q_weights = self.gguf_model.get_tensor(&q_proj_name)
                .ok_or_else(|| InferenceError::TensorNotFound(q_proj_name.clone()))?;
            // This is not a valid attention mechanism, just a placeholder for compute
            current_hidden_state = self.backend.matmul(&current_hidden_state, q_weights, false)?;
            // log::debug!("Hidden state shape after layer {} dummy attention: {:?}", i, current_hidden_state.shape);
        }
        // --- End Simplified Transformer Block Loop ---

        // 3. Final Normalization (if applicable)
        let final_norm_weights_name = "model.norm.weight"; // Example name
        if let Some(final_norm_weights) = self.gguf_model.get_tensor(final_norm_weights_name) {
            current_hidden_state = self.backend.rms_norm(&current_hidden_state, final_norm_weights, 1e-6f32)?;
        }

        // 4. LM Head: Project to vocabulary size to get logits
        //    Requires "lm_head.weight" or similar tensor.
        let lm_head_tensor_name = "lm_head.weight"; // Example name
        let lm_head_weights = self.gguf_model.get_tensor(lm_head_tensor_name)
            .ok_or_else(|| InferenceError::TensorNotFound(lm_head_tensor_name.to_string()))?;
        
        // We need to ensure current_hidden_state is 2D [seq_len, hidden_dim] for matmul with lm_head [vocab_size, hidden_dim]
        // The lm_head should be transposed for [seq_len, hidden_dim] @ [hidden_dim, vocab_size]
        // current_hidden_state = current_hidden_state.reshape(vec![token_ids.len(), hidden_dim]) ? // May need reshape
        let logits = self.backend.matmul(&current_hidden_state, lm_head_weights, true)?; // Assuming lm_head needs transpose
        // log::debug!("Logits shape: {:?}", logits.shape);

        // 5. Select the last token's logits for next token prediction
        //    Logits shape is typically [sequence_length, vocab_size].
        //    We need the logits for the *last* token in the input sequence.
        if logits.shape.len() != 2 || logits.shape[0] == 0 {
            return Err(InferenceError::LogicError("Logits tensor has unexpected shape".to_string()));
        }
        let last_token_logits_flat_index_start = (logits.shape[0] - 1) * logits.shape[1];
        let vocab_size = logits.shape[1];

        // For simplicity, just using argmax on the F32 logits of the last token.
        // A real implementation would need to correctly access the last row of the logits tensor.
        let predicted_token_id = match logits.dtype {
            DType::F32 => {
                if let TensorData::Owned(data_bytes) = &logits.data {
                    let logits_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(data_bytes.as_ptr() as *const f32, data_bytes.len() / 4)
                    };
                    
                    let last_logits_slice = &logits_f32[last_token_logits_flat_index_start .. last_token_logits_flat_index_start + vocab_size];
                    
                    last_logits_slice.iter().enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(index, _)| index as u32)
                        .ok_or_else(|| InferenceError::LogicError("Failed to find max logit".to_string()))?
                } else {
                    return Err(InferenceError::NotImplemented("Logit processing for non-owned tensor data".to_string()));
                }
            }
            _ => return Err(InferenceError::NotImplemented("Argmax for non-F32 logits".to_string())),
        };
        // log::info!("Predicted next token ID: {}", predicted_token_id);

        Ok(vec![predicted_token_id])
    }

    /// Convenience method to generate text (simple greedy approach for now).
    pub fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String, InferenceError> {
        let mut all_token_ids = self.tokenizer.tokenize(prompt)?;
        let mut generated_text = String::new();

        for _ in 0..max_new_tokens {
            // Create current prompt from all tokens so far
            // This is inefficient for long sequences; KV caching would be needed.
            let current_prompt_text = self.tokenizer.detokenize(&all_token_ids)?;
            let next_token_ids = self.forward(&current_prompt_text)?; // This will re-tokenize, also inefficient

            if let Some(next_token_id) = next_token_ids.first() {
                // TODO: Add check for EOS token ID here
                // if *next_token_id == self.tokenizer.eos_token_id() { break; }
                all_token_ids.push(*next_token_id);
                let next_token_text = self.tokenizer.detokenize(&[*next_token_id])?;
                generated_text.push_str(&next_token_text);
            } else {
                break; // No token predicted
            }
        }
        Ok(generated_text)
    }
} 