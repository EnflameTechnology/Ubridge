use core::ffi::{c_float, c_int, c_void};
use half::{bf16, f16};
use tops_backend::driv;
#[repr(C)]
pub struct dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
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

    // ─── Choreo production causal_conv1d (unified prefill + decode) ───
    pub fn causal_conv1d_fwd_choreo_bf16(
        x: *const bf16,
        weight: *const bf16,
        bias: *const bf16,
        conv_state: *mut bf16,
        cache_idx: *const i64,
        qsl: *const u32,
        out: *mut bf16,
        total_tokens: c_int,
        d_conv: c_int,
        batch: c_int,
        kernel_size: c_int,
        silu_activation: c_int,
        max_slots: c_int,
        stream: *const c_void,
    );

    // ─── GDN recurrence (prefill) ───
    pub fn gdn_recurrence_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_recurrence_f16(
        q: *const f16,
        k: *const f16,
        v: *const f16,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_recurrence_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut f32,
        bh: c_int,
        seq_len: c_int,
        k_dim: c_int,
        v_dim: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    // ─── GDN decode with slots ───
    pub fn gdn_decode_slots_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_decode_slots_f16(
        q: *const f16,
        k: *const f16,
        v: *const f16,
        g: *const f16,
        beta: *const f16,
        state: *mut f32,
        slots: *const i64,
        out: *mut f16,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_decode_slots_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        batch: c_int,
        heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    // ─── GDN decode with slots (GQA) ───
    pub fn gdn_decode_slots_gqa_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        qscale: *const f32,
        out: *mut f32,
        batch: c_int,
        num_v_heads: c_int,
        num_k_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    // ─── GDN decode L2Norm + recurrence + post fused (L2Norm + gating + GQA + RMSNorm + SiLU(z)) ───
    pub fn gdn_decode_recurrence_fused_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        a: *const bf16,
        b: *const bf16,
        a_log: *const f32,
        dt_bias: *const f32,
        z: *const bf16,
        state: *mut f32,
        slots: *const i64,
        norm_weight: *const f32,
        q_scale: c_float,
        out: *mut bf16,
        batch: c_int,
        nkh: c_int,
        nh: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    // ─── GDN varlen recurrence ───
    pub fn gdn_recurrence_varlen_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        slots: *const i64,
        out: *mut f32,
        cu_seqlens: *const u32,
        total_tokens: c_int,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_recurrence_varlen_f16(
        q: *const f16,
        k: *const f16,
        v: *const f16,
        g: *const f16,
        beta: *const f16,
        state: *mut f32,
        slots: *const i64,
        out: *mut f16,
        cu_seqlens: *const u32,
        total_tokens: c_int,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_recurrence_varlen_bf16(
        q: *const bf16,
        k: *const bf16,
        v: *const bf16,
        g: *const bf16,
        beta: *const bf16,
        state: *mut f32,
        slots: *const i64,
        out: *mut bf16,
        cu_seqlens: *const u32,
        total_tokens: c_int,
        batch: c_int,
        num_heads: c_int,
        k_dim: c_int,
        v_dim: c_int,
        max_slots: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );

    // ─── GDN fused gating (to be ported to Choreo) ───
    pub fn gdn_fused_gating_f32(
        a_log: *const f32,
        a: *const f32,
        b: *const f32,
        dt_bias: *const f32,
        g: *mut f32,
        beta: *mut f32,
        total: c_int,
        num_heads: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_fused_gating_f16(
        a_log: *const f32,
        a: *const f16,
        b: *const f16,
        dt_bias: *const f32,
        g: *mut f16,
        beta: *mut f16,
        total: c_int,
        num_heads: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_fused_gating_bf16(
        a_log: *const f32,
        a: *const bf16,
        b: *const bf16,
        dt_bias: *const f32,
        g: *mut bf16,
        beta: *mut bf16,
        total: c_int,
        num_heads: c_int,
        stream: *const c_void,
    );

    // ─── GDN L2 norm (to be ported to Choreo) ───
    pub fn gdn_l2_norm_f32(
        input: *const f32,
        output: *mut f32,
        rows: c_int,
        dim: c_int,
        eps: f32,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_l2_norm_f16(
        input: *const f16,
        output: *mut f16,
        rows: c_int,
        dim: c_int,
        eps: f32,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_l2_norm_bf16(
        input: *const bf16,
        output: *mut bf16,
        rows: c_int,
        dim: c_int,
        stream: *const c_void,
    );

    // ─── GDN gated RMSNorm + SiLU + mul (to be ported to Choreo) ───
    pub fn gdn_gated_rmsnorm_f32(
        x: *const f32,
        z: *const f32,
        gamma: *const f32,
        bias: *const f32,
        out: *mut f32,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: c_int,
        has_bias: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_gated_rmsnorm_f16(
        x: *const f16,
        z: *const f16,
        gamma: *const f32,
        bias: *const f32,
        out: *mut f16,
        rows: c_int,
        value_dim: c_int,
        group_size: c_int,
        eps: f32,
        per_group_weights: c_int,
        has_bias: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_gated_rmsnorm_bf16(
        x: *const bf16,
        z: *const bf16,
        gamma: *const f32,
        out: *mut bf16,
        rows: c_int,
        dim: c_int,
        stream: *const c_void,
    );

    // ─── GDN mamba scatter rows (to be ported to Choreo) ───
    pub fn gdn_mamba_scatter_f32(
        src: *const f32,
        dst: *mut f32,
        slots: *const i64,
        num_rows: c_int,
        row_elems: c_int,
        src_row_stride: c_int,
        dst_row_stride: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_mamba_scatter_f16(
        src: *const f16,
        dst: *mut f16,
        slots: *const i64,
        num_rows: c_int,
        row_elems: c_int,
        src_row_stride: c_int,
        dst_row_stride: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    pub fn gdn_mamba_scatter_bf16(
        src: *const bf16,
        dst: *mut bf16,
        slots: *const i64,
        num_rows: c_int,
        row_elems: c_int,
        src_row_stride: c_int,
        dst_row_stride: c_int,
        num_blocks: u32,
        dim_blocks: u32,
        stream: *const c_void,
    );
    
    pub fn reshape_and_cache_flash_host(
        dimBlocks: dim3,
        dimThreads: dim3,
        key: *const c_void,          // [num_tokens, num_heads, head_size]
        value: *const c_void,        // [num_tokens, num_heads, head_size]
        slot_mapping: *const c_void, // [num_tokens]
        key_cache: *const c_void,    // [num_blocks, block_size, num_heads, head_size]
        value_cache: *const c_void,  // [num_blocks, block_size, num_heads, head_size]
        dataType: i32,
        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        num_blocks: c_int,
        block_size: c_int,
        key_stride: c_int,
        value_stride: c_int,
        block_stride: c_int,
        page_stride: c_int,
        head_stride: c_int,
        stream: driv::topsStream_t,
    );
}

#[cfg(feature = "aten")]
extern "C" {
    // ─── topsaten MoE wrappers (optimized, graph-safe) ───
    pub fn topsaten_topk_softmax_f32(
        gating_output: *mut f32,
        topk_weights: *mut f32,
        topk_indices: *mut i32,
        token_expert_indices: *mut i32,
        num_tokens: c_int,
        num_experts: c_int,
        topk: c_int,
        norm_topk_prob: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_moe_align_block_size(
        sorted_token_ids: *mut i32,
        experts_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        topk_ids: *mut i32,
        num_tokens: c_int,
        topk: c_int,
        num_experts: c_int,
        block_size: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_fused_moe_gemm(
        c_ptr: *mut c_void,
        a_ptr: *mut c_void,
        b_ptr: *mut c_void,
        topk_weights: *mut f32,
        topk_ids: *mut i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_padded: *mut i32,
        mul_routed_weight: c_int,
        topk_val: c_int,
        block_size: c_int,
        num_tokens: c_int,
        hidden_dim: c_int,
        inter_dim: c_int,
        num_experts: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_release_graph_workspace_pool(stream: *mut c_void);

    // ─── topsaten optimized ops wrappers ───

    pub fn topsaten_rms_norm(
        output: *mut c_void,
        input: *const c_void,
        gamma: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        epsilon: f32,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_fused_add_rms_norm(
        input: *mut c_void,
        residual: *mut c_void,
        weight: *const c_void,
        num_tokens: c_int,
        hidden_size: c_int,
        epsilon: f32,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_silu_and_mul(
        output: *mut c_void,
        input: *const c_void,
        num_tokens: c_int,
        d: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_gelu_and_mul(
        output: *mut c_void,
        input: *const c_void,
        num_tokens: c_int,
        d: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_gelu_tanh_and_mul(
        output: *mut c_void,
        input: *const c_void,
        num_tokens: c_int,
        d: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_softmax(
        output: *mut c_void,
        input: *const c_void,
        n_rows: c_int,
        n_cols: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_matmul(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        batch: c_int,
        m: c_int,
        k: c_int,
        n: c_int,
        lhs_transpose: c_int,
        rhs_transpose: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_topk(
        output_value: *mut c_void,
        output_index: *mut c_void,
        input: *const c_void,
        n_rows: c_int,
        n_cols: c_int,
        k: c_int,
        is_largest: c_int,
        is_sorted: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_sort(
        output_sorted: *mut c_void,
        output_index: *mut c_void,
        input: *const c_void,
        n_rows: c_int,
        n_cols: c_int,
        descending: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_matmul_broadcast(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        batch: c_int,
        m: c_int,
        k: c_int,
        n: c_int,
        lhs_transpose: c_int,
        rhs_transpose: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_sum(
        output: *mut c_void,
        input: *const c_void,
        reduce_dim: c_int,
        num_dims: c_int,
        input_dims: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn ubridge_memcpy_d2d(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsaten_rotary_embedding(
        query: *mut c_void,
        key: *mut c_void,
        positions: *const c_void,
        cos_sin_cache: *const c_void,
        num_tokens: c_int,
        q_num_heads: c_int,
        k_num_heads: c_int,
        head_size: c_int,
        rot_dim: c_int,
        max_position: c_int,
        is_neox: c_int,
        qk_dtype_code: c_int,
        cache_dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    // ─── Unary ops ───
    pub fn topsaten_exp(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_log(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_sin(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_cos(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_tanh_op(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_silu(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_gelu(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_rsqrt(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_sqrt(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_neg(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_abs(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_sigmoid(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_reciprocal(
        out: *mut c_void,
        inp: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    // ─── Binary ops ───
    pub fn topsaten_add(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_sub(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_mul(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
    pub fn topsaten_div(
        out: *mut c_void,
        lhs: *const c_void,
        rhs: *const c_void,
        num_el: i64,
        ndim: c_int,
        shape: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    // ─── Efficient Attention ───
    pub fn topsaten_efficient_attention(
        out: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        bias: *const c_void,
        batch: c_int,
        q_seq: c_int,
        kv_seq: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        head_size: c_int,
        scale: f32,
        is_causal: c_int,
        has_bias: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    // ─── Flash Attention (topsfa) ───
    pub fn topsfa_flash_attn_fwd(
        out: *mut c_void,
        softmax_lse: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        alibi_slopes: *const c_void,
        batch: c_int,
        seqlen_q: c_int,
        seqlen_k: c_int,
        num_heads: c_int,
        num_heads_k: c_int,
        head_size: c_int,
        softmax_scale: f32,
        softcap: f32,
        is_causal: c_int,
        window_size_left: c_int,
        window_size_right: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsfa_flash_attn_varlen_fwd(
        out: *mut c_void,
        softmax_lse: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        cu_seqlens_q: *const c_void,
        cu_seqlens_k: *const c_void,
        block_table: *const c_void,
        alibi_slopes: *const c_void,
        total_q: c_int,
        total_k: c_int,
        batch: c_int,
        num_heads: c_int,
        num_heads_k: c_int,
        head_size: c_int,
        max_seqlen_q: c_int,
        max_seqlen_k: c_int,
        softmax_scale: f32,
        softcap: f32,
        is_causal: c_int,
        window_size_left: c_int,
        window_size_right: c_int,
        has_block_table: c_int,
        block_table_batch: c_int,
        block_table_max_blocks: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    pub fn topsfa_flash_attn_fwd_kvcache(
        out: *mut c_void,
        softmax_lse: *mut c_void,
        q: *const c_void,
        kcache: *const c_void,
        vcache: *const c_void,
        seqlens_k: *const c_void,
        block_table: *const c_void,
        alibi_slopes: *const c_void,
        batch: c_int,
        seqlen_q: c_int,
        num_heads: c_int,
        num_heads_k: c_int,
        head_size: c_int,
        seqlen_k: c_int,
        page_block_size: c_int,
        num_blocks: c_int,
        max_num_blocks_per_seq: c_int,
        softmax_scale: f32,
        softcap: f32,
        is_causal: c_int,
        window_size_left: c_int,
        window_size_right: c_int,
        is_rotary_interleaved: c_int,
        num_splits: c_int,
        has_block_table: c_int,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;

    /// Strided / general copy via topsatenCopy (large-tensor fallback for ucopy_*).
    pub fn topsaten_copy(
        out: *mut c_void,
        inp: *const c_void,
        ndim: c_int,
        out_shape: *const i64,
        out_strides: *const i64,
        in_shape: *const i64,
        in_strides: *const i64,
        dtype_code: c_int,
        stream: *const c_void,
    ) -> c_int;
}
