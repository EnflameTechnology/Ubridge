use core::ffi::{c_int, c_long, c_void};
use half::{bf16, f16};

extern "C" {
    pub fn topk_f32(
        input: *mut f32,
        output: *mut f32,
        indices: *mut u32,
        workspace: *mut c_void,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        axis: c_int,
        k: c_int,
        stream: *const c_void,
    );

    pub fn topk_f16(
        input: *mut f16,
        output: *mut f16,
        indices: *mut u32,
        workspace: *mut c_void,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        axis: c_int,
        k: c_int,
        stream: *const c_void,
    );

    pub fn topk_bf16(
        input: *mut bf16,
        output: *mut bf16,
        indices: *mut u32,
        workspace: *mut c_void,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        axis: c_int,
        k: c_int,
        stream: *const c_void,
    );
}
