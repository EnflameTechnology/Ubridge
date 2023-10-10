// extern crate ndarray;
// extern crate ndarray_linalg;
// extern crate ndarray_rand;

pub mod device_executor;
pub mod device_opcode;
pub mod device_tensor;
pub mod gcu_device;
pub mod gcu_slice;
pub mod device_ptr;
pub mod gcu_launch;
pub mod tests;
/// Prelude module for users to import
pub mod prelude {
    // prelude
    pub use crate::device_executor::*;
    pub use crate::device_tensor::*;
    pub use crate::device_opcode::*;
    pub use crate::gcu_device::*;
    pub use crate::device_ptr::*;
    pub use crate::gcu_slice::*;
    pub use crate::gcu_launch::*;
    pub use crate::tests::*;

    #[cfg(feature = "tops_backend")]
    pub use tops_backend::memory::CopyDestination;

    #[cfg(feature = "cuda_backend")]
    pub use cuda_backend::memory::CopyDestination;
}

pub const AFFINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/affine.topsfb");
pub const BINARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/binary.topsfb");
pub const CAST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/cast.topsfb");
pub const CONV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/conv.topsfb");
pub const FILL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/fill.topsfb");
pub const INDEXING: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/indexing.topsfb");
pub const REDUCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/reduce.topsfb");
pub const TERNARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/ternary.topsfb");
pub const UNARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/unary.topsfb");
pub const MATMUL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/matmul.topsfb");

pub const TRANSPOSE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/transpose.topsfb");
pub const DOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/dot.topsfb");
pub const DOTLLM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/dotllm.topsfb");

