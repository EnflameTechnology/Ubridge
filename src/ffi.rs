use core::ffi::{c_int, c_void};
use half::{bf16, f16};

#[repr(C)]
pub struct SelectiveScanParas {
    pub b: i64,
    pub h: i64,
    pub h_align: i64,
    pub l: i64,
    pub d: i64,
    pub l2_mem_used: i64,
    pub subk_size: i64,
    pub l1_mem_used: i64,
    pub data_type: c_int,
    pub bpe: c_int,
    pub has_bias: bool,
    pub has_softplus: bool,
    pub has_pre_state: bool,
    pub return_last_state: bool,
}

#[repr(C)]
pub struct CausalConv1dParams {
    pub dim: c_int,
    pub batch: c_int,
    pub num_cache_lines: c_int,
    pub kernel_width: c_int,
    pub state_len: c_int,
    pub stride_x_token: c_int,
    pub stride_w_dim: c_int,
    pub stride_istate_seq: c_int,
    pub stride_istate_token: c_int,
    pub pad_slot_id: c_int,
    pub has_bias: c_int,
    pub silu_activation: c_int,
    pub block_n: c_int,
}

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

    pub fn mask_f32(
        input: *mut f32,
        v: f32,
        output: *mut u32,
        batch: i32,
        dim_size: i32,
        stream: *const c_void,
    ) -> u32;

    pub fn mask_i32(
        input: *mut i32,
        v: i32,
        output: *mut u32,
        batch: i32,
        dim_size: i32,
        stream: *const c_void,
    ) -> u32;

    pub fn mask_u32(
        input: *mut u32,
        v: u32,
        output: *mut u32,
        batch: i32,
        dim_size: i32,
        stream: *const c_void,
    ) -> u32;

    pub fn indexed_moe_f16(
        input: *mut f16,
        w: *mut f16,
        out: *mut f16,
        idx: *mut u32,
        N: c_int,
        K: c_int,
        M: c_int,
        batch: c_int,
        topk: c_int,
        num_experts: c_int,
        stream: *const c_void,
    );

    pub fn indexed_moe_bf16(
        input: *mut bf16,
        w: *mut bf16,
        out: *mut bf16,
        idx: *mut u32,
        N: c_int,
        K: c_int,
        M: c_int,
        batch: c_int,
        topk: c_int,
        num_experts: c_int,
        stream: *const c_void,
    );

    pub fn moe_align_block_size(
        sorted_ids: *mut i32,
        experts_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        topk_ids: *const i32,
        real_token_num: *const i32,
        expert_map: *const i32,
        num_experts: c_int,
        block_size: c_int,
        token_num: c_int,
        topk: c_int,
        stream: *const c_void,
    );

    pub fn invoke_fused_moe_f16(
        c: *mut f16,
        a: *mut f16,
        b: *mut f16,
        bias: *mut f16,
        topk_weights: *mut f32,
        sorted_ids: *mut i32,
        experts_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        m: c_int,
        k: c_int,
        n: c_int,
        e: c_int,
        topk_ids_size: c_int,
        top_k: c_int,
        block_size: c_int,
        mul_routed_weight: c_int,
        bias_flag: c_int,
        stream: *const c_void,
    );

    pub fn invoke_fused_moe_bf16(
        c: *mut bf16,
        a: *mut bf16,
        b: *mut bf16,
        bias: *mut bf16,
        topk_weights: *mut f32,
        sorted_ids: *mut i32,
        experts_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        m: c_int,
        k: c_int,
        n: c_int,
        e: c_int,
        topk_ids_size: c_int,
        top_k: c_int,
        block_size: c_int,
        mul_routed_weight: c_int,
        bias_flag: c_int,
        stream: *const c_void,
    );

    pub fn topk_softmax_f32(
        input: *mut f32,
        output: *mut f32,
        index: *mut i32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        norm_topk_prob: c_int,
        stream: *const c_void,
    );

    pub fn topk_softmax_f16(
        input: *mut f16,
        output: *mut f16,
        index: *mut i32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        norm_topk_prob: c_int,
        stream: *const c_void,
    );

    pub fn topk_softmax_bf16(
        input: *mut bf16,
        output: *mut bf16,
        index: *mut i32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        norm_topk_prob: c_int,
        stream: *const c_void,
    );

    pub fn causal_conv1d_fwd_f32(
        x_ptr: *mut f32,
        w_ptr: *mut f32,
        bias_ptr: *mut f32,
        conv_states_ptr: *mut f32,
        cache_indices_ptr: *mut i32,
        has_initial_states_ptr: *mut i8,
        query_start_loc_ptr: *mut i32,
        o_ptr: *mut f32,
        params: *mut CausalConv1dParams,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    pub fn causal_conv1d_fwd_f16(
        x_ptr: *mut f16,
        w_ptr: *mut f16,
        bias_ptr: *mut f16,
        conv_states_ptr: *mut f16,
        cache_indices_ptr: *mut i32,
        has_initial_states_ptr: *mut i8,
        query_start_loc_ptr: *mut i32,
        o_ptr: *mut f16,
        params: *mut CausalConv1dParams,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    pub fn causal_conv1d_fwd_bf16(
        x_ptr: *mut bf16,
        w_ptr: *mut bf16,
        bias_ptr: *mut bf16,
        conv_states_ptr: *mut bf16,
        cache_indices_ptr: *mut i32,
        has_initial_states_ptr: *mut i8,
        query_start_loc_ptr: *mut i32,
        o_ptr: *mut bf16,
        params: *mut CausalConv1dParams,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
}
