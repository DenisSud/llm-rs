## Rust LLM Inference Engine: Project Plan

**Goal**: Build a cross-platform, Rust-based LLM inference engine leveraging \[wgpu] for GPU acceleration. This plan outlines crate responsibilities, specifications, implementation tasks, and integration points for a team of developers.

---

### Workspace Overview

```
rust-llm-inference/           # Workspace root
├── Cargo.toml                # Workspace definition
├── README.md                 # High‑level overview & getting started
├── docs/                     # Design docs & roadmaps
│   ├── architecture.md       # Deep dive on design decisions
│   └── ROADMAP.md            # Milestones & schedule
└── crates/                   # Core crates
    ├── gguf-parser          # GGUF model file parsing
    ├── core                 # Tensor and data-type definitions
    ├── backend-api          # Trait definitions for compute backends
    ├── cpu-backend          # CPU reference implementation
    ├── wgpu-backend         # GPU acceleration via wgpu + WGSL shaders
    ├── tokenizer            # Text tokenization utilities
    ├── inference            # High‑level Model API & pipelines
    └── cli                  # Command‑line interface
```

Each crate is a standalone library (except `cli`), with minimal dependencies:

* **Dependency flow**: `cli` → `inference` → \[`gguf-parser`, `core`, `backend-api`, `tokenizer`] → \[`cpu-backend` | `wgpu-backend`]
* Shared utilities (logging, errors) can be added under `core` or a new `utils` crate if needed.

---

## Crate Specifications

### 1. `gguf-parser`

**Purpose**: Read `.gguf` model files, validate headers, extract metadata and weight tensors.

**Tasks**:

* Implement header struct and magic‑byte checks.
* Parse metadata entries: model architecture, tensor names, shapes, dtypes, tokenizer links.
* Load raw tensor data using optional `mmap` feature (fallback to streamed reads).
* Expose:

  ```rust
  pub struct GgufModel { /* metadata + Vec<Tensor> */ }
  pub fn load(path: &Path) -> Result<GgufModel, Error>;
  ```

  * `GgufModel` holds a list of `core::Tensor` and metadata in a typed struct.

### 2. `core`

**Purpose**: Define fundamental data structures: `Tensor`, `DType`, error types, and utilities.

**Tasks**:

* `enum DType { F32, F16, BF16, I8, U8 }`
* `struct Tensor { shape: Vec<usize>, dtype: DType, data: Vec<u8> }`
* Implement shape utilities (e.g., `num_elements`), memory-size calculations.
* Error and logging utilities (or extract into `utils`).

### 3. `backend-api`

**Purpose**: Abstract compute operations via a trait to allow swapping CPU/GPU implementations.

**Tasks**:

* Define trait:

  ```rust
  pub trait Backend {
      fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor;
      fn softmax(&self, input: &Tensor) -> Tensor;
      fn layer_norm(&self, input: &Tensor, gamma: &Tensor, beta: &Tensor) -> Tensor;
      // more ops as needed
  }
  ```
* Provide `BackendError` type.

### 4. `cpu-backend`

**Purpose**: Reference CPU implementation (rayon + BLAS or pure Rust) for correctness and fallback.

**Tasks**:

* Implement `Backend` trait methods with multi-threading.
* Unit tests comparing known results (e.g., small matrices, softmax outputs).
* Benchmark suite for matmul and softmax.

### 5. `wgpu-backend`

**Purpose**: Leverage `wgpu` and WGSL shaders to accelerate compute across platforms (Vulkan, Metal, DX12).

**Tasks**:

* Initialize WGPU instance, choose adapter and device with required features.
* Shader modules (`shaders/`) implementing `matmul`, `softmax`, `layer_norm`, etc.
* Write Rust wrappers to upload `Tensor` buffers to GPU, dispatch compute, and read back results.
* Ensure synchronization and proper buffer layouts.
* Integration tests vs. CPU backend for numeric equivalence.

### 6. `tokenizer`

**Purpose**: Text-to-token and token-to-text conversions using pre-trained vocab files.

**Tasks**:

* Implement loading of BPE/vocab files (e.g., from Hugging Face format).
* `fn tokenize(text: &str) -> Vec<u32>` and `fn detokenize(ids: &[u32]) -> String`.
* Unit tests for known sentences.

### 7. `inference`

**Purpose**: High‑level API combining parser, tokenizer, and backend into a model pipeline.

**Tasks**:

* `struct Model { gguf: GgufModel, backend: Box<dyn Backend>, tokenizer: Tokenizer }`
* `fn load(path: &Path, backend: BackendChoice) -> Result<Model, Error>`
* `fn forward(&self, prompt: &str) -> Result<String, Error>`:

  * Tokenize prompt → sequence of IDs
  * For each transformer layer:

    * Load weight tensors from `gguf` into `Tensor`
    * Call `backend.matmul`, `backend.layer_norm`, etc.
  * Decode output token IDs to text
* Support streaming generation (optional milestone).

### 8. `cli`

**Purpose**: Expose command‑line interface for quick testing and demos.

**Tasks**:

* Subcommands: `run` (single-inference), `benchmark`, `info` (model metadata).
* Parse arguments with `clap` or `structopt`.
* Hook into `inference::Model`.

---

## Integration & CI

* **Workspace Cargo.toml** ensures crate versions stay in sync.
* **Continuous Integration** (GitHub Actions):

  * Build and test all crates on Linux, macOS, Windows.
  * GPU tests: optionally run with `wgpu` on supported platforms or mock.
  * Code formatting (`rustfmt`) and lints (`clippy`).
* **Documentation**: Auto‑generate docs via `cargo doc` and host on GitHub Pages.

---

## Milestones & Timeline

| Milestone                        | Duration | Owners        |
| -------------------------------- | -------- | ------------- |
| Workspace & core API definitions | 1 week   | Core Team     |
| GGUF parser implementation       | 1 week   | Parser Team   |
| CPU backend + unit tests         | 1 week   | Backend Team  |
| Tokenizer integration            | 1 week   | NLP Team      |
| WGPU backend prototype           | 2 weeks  | GPU Team      |
| Inference orchestration & CLI    | 2 weeks  | API Team      |
| End‑to‑end tests & benchmarking  | 1 week   | QA Team       |
| Documentation & release v0.1.0   | 1 week   | Documentation |
