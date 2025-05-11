use std::path::PathBuf;
use clap::Parser;
use inference::{Model, InferenceError};
use backend_api::BackendChoice; // Import BackendChoice

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    /// Run inference on a prompt
    Run {
        /// Path to the GGUF model file
        #[clap(short, long, value_parser)]
        model: PathBuf,

        /// Path to the tokenizer vocabulary file (e.g., vocab.json or tokenizer.model)
        #[clap(short, long, value_parser)]
        tokenizer_vocab: PathBuf,

        /// Prompt to run inference on
        #[clap(short, long)]
        prompt: String,

        /// Maximum number of new tokens to generate
        #[clap(long, default_value_t = 64)]
        max_tokens: usize,

        /// Backend to use for computation
        #[clap(short, long, value_enum, default_value_t = CliBackendChoice::Cpu)]
        backend: CliBackendChoice,
    },
    /// Display model information (Not yet implemented)
    Info {
        /// Path to the GGUF model file
        #[clap(short, long, value_parser)]
        model: PathBuf,
    },
    // Benchmark command could be added later
}

// Clap anscilliary enum for BackendChoice
#[derive(clap::ValueEnum, Clone, Debug, Copy)]
enum CliBackendChoice {
    Cpu,
    Wgpu,
}

impl From<CliBackendChoice> for BackendChoice {
    fn from(value: CliBackendChoice) -> Self {
        match value {
            CliBackendChoice::Cpu => BackendChoice::Cpu,
            CliBackendChoice::Wgpu => BackendChoice::Wgpu,
        }
    }
}


fn main() -> Result<(), CliError> {
    // TODO: Initialize logger (e.g., env_logger::init();)
    let args = Args::parse();

    match args.command {
        Commands::Run { model: model_path, tokenizer_vocab: vocab_path, prompt, max_tokens, backend } => {
            println!("Loading model from: {:?}", model_path);
            println!("Using tokenizer vocab: {:?}", vocab_path);
            println!("Selected backend: {:?}", backend);
            
            let model = Model::load(&model_path, &vocab_path, backend.into())?;
            println!("Model loaded successfully.");

            println!("Prompt: {}", prompt);
            // let next_token_ids = model.forward(&prompt)?;
            // println!("Predicted next token ID(s): {:?}", next_token_ids);
            // // To see text, you'd need a tokenizer.detokenize method
            // if let Some(first_id) = next_token_ids.first() {
            //     // This requires the tokenizer to be accessible or for detokenize to be part of Model.
            //     // For now, we'll just print the ID.
            //     // let next_token_text = tokenizer.detokenize(&[*first_id])?;
            //     // println!("Predicted next token text (approx): {}", next_token_text);
            // }

            match model.generate(&prompt, max_tokens) {
                Ok(generated_text) => {
                    println!("Generated text:\n{}", generated_text);
                }
                Err(e) => {
                    eprintln!("Error during generation: {}", e);
                }
            }
        }
        Commands::Info { model: model_path } => {
            println!("Displaying info for model: {:?} (Not Implemented Yet)", model_path);
            // TODO: Implement GGUF info display by loading GgufModel and printing metadata.
        }
    }

    Ok(())
}

#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("Inference error: {0}")]
    Inference(#[from] InferenceError),
    // #[error("Tokenizer error: {0}")] // Already covered by InferenceError
    // Tokenizer(#[from] tokenizer::TokenizerError),
} 