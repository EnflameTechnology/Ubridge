use core::ffi::{c_int, c_long, c_void};
use half::{bf16, f16};

extern "C" {
    pub fn topk_f32(
        input: *mut f32,
        output: *mut f32,
        indices: *mut u32,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        k: c_int,
        stream: *const c_void,
    );

    pub fn topk_f16(
        input: *mut f16,
        output: *mut f16,
        indices: *mut u32,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        k: c_int,
        stream: *const c_void,
    );

    pub fn topk_bf16(
        input: *mut bf16,
        output: *mut bf16,
        indices: *mut u32,
        dim0: c_int,
        dim1: c_int,
        dim2: c_int,
        k: c_int,
        stream: *const c_void,
    );

    pub fn moe_f16(
        y: *mut f16,
        e_out: *mut f16,
        w: *mut f32,
        idx: *mut u32,
        top: *mut u32,
        N: c_int,
        K: c_int,
        M: c_int,
        topk: c_int,
        stream: *const c_void,
    );

    pub fn moe_bf16(
        y: *mut bf16,
        e_out: *mut bf16,
        w: *mut f32,
        idx: *mut u32,
        top: *mut u32,
        N: c_int,
        K: c_int,
        M: c_int,
        topk: c_int,
        stream: *const c_void,
    );

    pub fn dequant_f16(
        out: *mut f16,
        rhs: *mut u8,
        scale: *mut f16,
        zeros: *mut f16,
        K: c_int, 
        N: c_int,
        weight_transpose: c_int,
        group_size: c_int,
        stream: *const c_void,
    );

    pub fn dequant_bf16(
        out: *mut bf16,
        rhs: *mut u8,
        scale: *mut bf16,
        zeros: *mut bf16,
        K: c_int,
        N: c_int,
        weight_transpose: c_int,
        group_size: c_int,
        stream: *const c_void,
    );
}
