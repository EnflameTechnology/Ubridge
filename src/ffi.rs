
use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub fn topk_f32(
        input: *const c_void,
        output: *mut c_void,
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
        input: *const c_void,
        output: *mut c_void,
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
        input: *const c_void,
        output: *mut c_void,
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
