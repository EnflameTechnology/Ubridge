// extern crate ndarray;
// extern crate ndarray_linalg;
// extern crate ndarray_rand;

pub mod device_executor;
pub mod device_opcode;
pub mod device_tensor;

/// Prelude module for users to import
pub mod prelude {
    // prelude
    pub use crate::device_executor::*;
    pub use crate::device_tensor::*;
    pub use crate::device_opcode::*;
    #[cfg(feature = "tops_backend")]
    pub use tops_backend::memory::CopyDestination;

    #[cfg(feature = "cuda_backend")]
    pub use cuda_backend::memory::CopyDestination;
}