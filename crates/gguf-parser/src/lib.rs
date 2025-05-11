//! A minimal GGUF loader: mmap the file, parse header & index, then materialize Tensors.

mod header;
mod io;
mod tensor;

pub use header::GgufHeader;
pub use io::GgufModel;
pub use tensor::{GgufEntry, GgufTensor};
