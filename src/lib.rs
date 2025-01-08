/*
* Copyright 2021-2024 Enflame. All Rights Reserved.

* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

pub mod device_executor;
pub mod device_opcode;
pub mod device_ptr;
pub mod device_tensor;
pub mod gcu_device;
pub mod gcu_launch;
pub mod gcu_slice;
pub mod gemm_tuner;
pub mod tests;
#[cfg(feature = "eccl")]
pub mod eccl;
#[cfg(feature = "eccl")]
pub mod eccllib;
/// Prelude module for users to import
pub mod prelude {
    // prelude
    pub use crate::device_executor::*;
    pub use crate::device_opcode::*;
    pub use crate::device_ptr::*;
    pub use crate::device_tensor::*;
    pub use crate::gcu_device::*;
    pub use crate::gcu_launch::*;
    pub use crate::gcu_slice::*;
    pub use crate::gemm_tuner::*;
    pub use crate::tests::*;

    #[cfg(feature = "tops_backend")]
    pub use tops_backend::memory::CopyDestination;
    #[cfg(feature = "tops_backend")]
    pub use uhal::stream::StreamTrait;

    #[cfg(feature = "cuda_backend")]
    pub use cuda_backend::memory::CopyDestination;

    #[cfg(feature = "eccl")]
    pub use crate::eccl::*;
}

// pub const AFFINE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/affine.topsfb");
// pub const BINARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/binary.topsfb");
// pub const CAST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/cast.topsfb");
// pub const CONV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/conv.topsfb");
// pub const FILL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/fill.topsfb");
// pub const INDEXING: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/indexing.topsfb");
// pub const REDUCE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/reduce.topsfb");
// pub const TERNARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/ternary.topsfb");
// pub const UNARY: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/unary.topsfb");
// pub const MATMUL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/matmul.topsfb");
// pub const KCCONCAT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/kvconcat.topsfb");

pub const AFFINE: &str = "affine";
pub const BINARY: &str = "binary";
pub const CAST: &str = "cast";
pub const CONV: &str = "conv";
pub const FILL: &str = "fill";
pub const INDEXING: &str = "indexing";
pub const REDUCE: &str = "reduce";
pub const TERNARY: &str = "ternary";
pub const UNARY: &str = "unary";
pub const MATMUL: &str = "matmul";
pub const KCCONCAT: &str = "kvconcat";
pub const FILLCOPY: &str = "copy2d";
pub const EMBEDDING: &str = "embedding";
pub const QUANTIZED: &str = "quant";

// pub const TRANSPOSE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/transpose.topsfb");
// pub const DOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/dot.topsfb");
// pub const DOTLLM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/dotllm.topsfb");
// pub const GEMM: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernels/gemm.topsfb");

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DATATYPE {
    DataNone = -1,
    DataI8 = 0,
    DataU8,
    DataI16,
    DataU16,
    DataFp16,
    DataBf16,
    DataI32,
    DataU32,
    DataFp32,
    DataEf32,
    DataTf32,
    DataI64,
    DataU64,
    DataF64,
    DataPred,
    DataI4,
}
