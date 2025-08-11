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
*
* @file    gemm_tuner.rs
* @brief
*
* @author  Guoqing Bao
* @date    2023-11-15 - 2024-01-09
* @version V0.1
* @par     Copyright (c) Enflame Tech Company.
* @par     History: gemm tuner with cache optimization
* @par     Comments: a gemm tuner bought from TopsOp and modified for candle-gcu
*/
#![allow(warnings, unused)]
use std::{collections::HashMap, sync::Mutex};

pub use crate::DATATYPE;
pub use cust_core::_hidden::DeviceCopy;
use std::cell::UnsafeCell;
// CeilDiv: ceil division
fn ceil_div<T: num_traits::One>(x: T, y: T) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + Copy,
{
    (x + y - T::one()) / y
}

// AlignUp: align to a multiple of rhs no less than lhs
fn align_up<T>(x: T, y: T) -> T
where
    T: std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::Sub<Output = T>
        + std::ops::Div<Output = T>
        + num_traits::One
        + Copy,
{
    ceil_div(x, y) * y
}

// AlignDown: align to a multiple of rhs no more than lhs
fn align_down<T>(x: T, y: T) -> T
where
    T: std::ops::Div<Output = T> + std::ops::Mul<Output = T> + Copy,
{
    (x / y) * y
}
// Gemm Begin

macro_rules! init_split {
    ($info: ident, $tune: ident) => {
        $tune.csb_batch = 1;
        $tune.sip_batch = 1;
        $tune.lhs_csb_m = 0;
        $tune.lhs_csb_k = 0;
        $tune.rhs_csb_k = 0;
        $tune.rhs_csb_n = 0;
        let sip_m = 0;
        let sip_k = 0;
        let sip_n = 0;
        let lhs_tranpose = $info.transa;
        let rhs_tranpose = $info.transb;
        let out_tranpose = false;
        let batch_multicore = false;
        let lhs_multicore = false;
        let rhs_multicore = false;
        let cdma_lhs_pingpong = false;
        let cdma_rhs_pingpong = false;
        let sdma_lhs_pingpong = false;
        let sdma_rhs_pingpong = false;
        let rhs_repeatcopy = false;
        let M = $info.M;
        let N = $info.N;
        let K = $info.K;
        let B = $info.batch;
    };
}

macro_rules! set_split_option {
    ($tune: ident, $batch_multicore:expr, $lhs_multicore:expr, $rhs_multicore:expr, $cdma_lhs_pingpong:expr, $cdma_rhs_pingpong:expr, $sdma_lhs_pingpong:expr, $sdma_rhs_pingpong:expr, $rhs_repeatcopy:expr) => {{
        $tune.batch_multicore = $batch_multicore;
        $tune.lhs_multicore = $lhs_multicore;
        $tune.rhs_multicore = $rhs_multicore;
        $tune.cdma_lhs_pingpong = $cdma_lhs_pingpong;
        $tune.cdma_rhs_pingpong = $cdma_rhs_pingpong;
        $tune.sdma_lhs_pingpong = $sdma_lhs_pingpong;
        $tune.sdma_rhs_pingpong = $sdma_rhs_pingpong;
        $tune.rhs_repeatcopy = $rhs_repeatcopy;
    }};
}

#[allow(non_snake_case)]
pub struct AtenGemmInfo {
    pub data_type: DATATYPE,
    pub weight_type: DATATYPE,
    pub out_data_type: DATATYPE,
    pub is_batch: bool,
    pub batch: i64,
    pub M: i64,
    pub K: i64,
    pub N: i64,
    pub transa: bool,
    pub transb: bool,
}

impl AtenGemmInfo {
    pub fn new(
        datatype: DATATYPE,
        weight_type: DATATYPE,
        batch: usize,
        M: usize,
        K: usize,
        N: usize,
        rhs_trans: i32,
    ) -> AtenGemmInfo {
        AtenGemmInfo {
            data_type: datatype,
            weight_type,
            out_data_type: datatype,
            is_batch: batch > 1,
            batch: batch as i64,
            M: M as i64,
            K: K as i64,
            N: N as i64,
            transa: false,
            transb: rhs_trans > 0,
        }
    }
}

impl Default for AtenGemmInfo {
    fn default() -> AtenGemmInfo {
        AtenGemmInfo {
            data_type: DATATYPE::DataFp32,
            weight_type: DATATYPE::DataFp32,
            out_data_type: DATATYPE::DataFp32,
            is_batch: false,
            batch: 1,
            M: 1,
            K: 4096,
            N: 4096,
            transa: false,
            transb: true,
        }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct AtenGemmTune {
    csb_batch: i64,
    sip_batch: i64,
    lhs_csb_m: i64,
    lhs_csb_k: i64,
    rhs_csb_k: i64,
    rhs_csb_n: i64,

    sip_m: i64,
    sip_k: i64,
    sip_n: i64,

    batch_multicore: bool,
    lhs_multicore: bool,
    rhs_multicore: bool,
    cdma_lhs_pingpong: bool,
    cdma_rhs_pingpong: bool,
    sdma_lhs_pingpong: bool,
    sdma_rhs_pingpong: bool,
    rhs_repeatcopy: bool,

    lhs_tranpose: bool,
    rhs_tranpose: bool,
    out_tranpose: bool,

    pattern_type: i32, // 1: lcache, 2: batch, 3: general
}

impl Default for AtenGemmTune {
    fn default() -> AtenGemmTune {
        AtenGemmTune {
            csb_batch: 1,
            sip_batch: 1,
            lhs_csb_m: 0,
            lhs_csb_k: 0,
            rhs_csb_k: 0,
            rhs_csb_n: 0,

            sip_m: 0,
            sip_k: 0,
            sip_n: 0,

            batch_multicore: false,
            lhs_multicore: false,
            rhs_multicore: false,
            cdma_lhs_pingpong: false,
            cdma_rhs_pingpong: false,
            sdma_lhs_pingpong: false,
            sdma_rhs_pingpong: false,
            rhs_repeatcopy: false,

            lhs_tranpose: false,
            rhs_tranpose: false,
            out_tranpose: false,

            pattern_type: 3,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GEMM_OP_PARAS {
    pub input_dtype: i32, // 0
    pub output_dtype: i32,
    pub csb_batch: i32,
    pub sip_batch: i32,
    pub lhs_csb_k: i32,
    pub rhs_csb_k: i32, // 5
    pub lhs_csb_m: i32,
    pub rhs_csb_n: i32,
    pub sip_m: i32,
    pub sip_k: i32,
    pub sip_n: i32, // 10
    pub batch_multicore: i32,
    pub lhs_multicore: i32,
    pub rhs_multicore: i32,
    pub lhs_pingpong: i32,
    pub rhs_pingpong: i32,
    pub sdma_lhs_pingpong: i32, // 15
    pub sdma_rhs_pingpong: i32,
    pub rhs_repeat_copy: i32,
    pub lhs_transpose: i32,
    pub rhs_transpose: i32,
    pub out_transpose: i32, // 20
    pub alpha: f32,
    pub beta: f32,
    pub addmm_beta: f32,
    pub coef: f32,
    pub act_mode: f32,
    pub bias: i32, // 25
    pub act_en: i32,
    pub input_batch: i32,
    pub input_m: i32,
    pub input_k: i32,
    pub input_n: i32,
}

// unsafe impl DeviceCopy for GEMM_OP_PARAS {
//     fn as_kernel_param(&self) -> *mut std::ffi::c_void {
//         self as *const Self as *mut _
//     }
// }

impl GEMM_OP_PARAS {
    pub fn new(info: &AtenGemmInfo, tune: &AtenGemmTune) -> GEMM_OP_PARAS {
        GEMM_OP_PARAS {
            input_dtype: info.data_type as i32, // 0
            output_dtype: info.out_data_type as i32,
            csb_batch: tune.csb_batch as i32,
            sip_batch: tune.sip_batch as i32,
            lhs_csb_k: tune.lhs_csb_k as i32,
            rhs_csb_k: tune.rhs_csb_k as i32, // 5
            lhs_csb_m: tune.lhs_csb_m as i32,
            rhs_csb_n: tune.rhs_csb_n as i32,
            sip_m: tune.sip_m as i32,
            sip_k: tune.sip_k as i32,
            sip_n: tune.sip_n as i32, // 10
            batch_multicore: tune.batch_multicore as i32,
            lhs_multicore: tune.lhs_multicore as i32,
            rhs_multicore: tune.rhs_multicore as i32,
            lhs_pingpong: tune.cdma_lhs_pingpong as i32,
            rhs_pingpong: tune.cdma_rhs_pingpong as i32,
            sdma_lhs_pingpong: tune.sdma_lhs_pingpong as i32, // 15
            sdma_rhs_pingpong: tune.sdma_rhs_pingpong as i32,
            rhs_repeat_copy: tune.rhs_repeatcopy as i32,
            lhs_transpose: info.transa as i32,
            rhs_transpose: info.transb as i32,
            out_transpose: tune.out_tranpose as i32, // 20
            alpha: 1.0,
            beta: 0.0,
            addmm_beta: 0.0,
            coef: 0.0,
            act_mode: 0.0, //TOPSOP_ACTIVATION_NONE,
            bias: 0,
            act_en: 0,
            input_batch: info.batch as i32,
            input_m: info.M as i32,
            input_k: info.K as i32,
            input_n: info.N as i32,
        }
    }
}

// #[derive(Clone)]
pub struct AtenGemmTuner {
    _op_name: String,
    tuned_map: Mutex<UnsafeCell<HashMap<String, GEMM_OP_PARAS>>>,
}

impl Default for AtenGemmTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl AtenGemmTuner {
    pub fn new() -> Self {
        AtenGemmTuner {
            _op_name: String::new(),
            tuned_map: Mutex::new(HashMap::<String, GEMM_OP_PARAS>::new().into()),
        }
    }

    pub unsafe fn tuner(&self, info: &AtenGemmInfo) -> &GEMM_OP_PARAS {
        let pattern = format!(
            "t{:?}_b{}_m{}_k{}_n{}",
            info.data_type, info.batch, info.M, info.K, info.N
        );
        let tuned_map = self.tuned_map.lock().unwrap();
        let mutmap = match tuned_map.get().as_mut() {
            Some(_b) => _b,
            _ => {
                panic!("error")
            }
        };

        if mutmap.contains_key(&pattern) {
            &mutmap[&pattern.clone()]
        } else {
            let mut tune = AtenGemmTune::default();
            if info.data_type == info.out_data_type {
                match info.data_type {
                    DATATYPE::DataFp32 => {
                        self.tuner_sgemm_f32(info, &mut tune);
                    }
                    _ => {
                        self.tuner_hgemm_half(info, &mut tune);
                    }
                }
            } else {
                // handle the case when data_type is not equal to out_data_type
            }
            let param = GEMM_OP_PARAS::new(info, &tune);
            mutmap.insert(pattern.clone(), param);
            &mutmap[&pattern]
        }
    }

    fn tuner_sgemm_f32(&self, info: &AtenGemmInfo, tune: &mut AtenGemmTune) -> i32 {
        init_split!(info, tune);

        let sum_mem = |m: i64, n: i64, k: i64, bpe: i64, bias: i64| -> i64 {
            (2 * m * k + m * n + 2 * n * k + bias) * bpe * 2
        };

        let sum_va_mem = |m: i64, n: i64, bpe: i64| -> i64 { (m * n * 2) * bpe * 2 };

        const UNIT_SIP_M: i64 = 32;
        const UNIT_SIP_N: i64 = 64;
        const UNIT_SIP_K: i64 = 32;
        const BPE: i64 = 4;
        let mut sip_cnt = 6;
        let mut l1_mem = 0;
        let mut va_mem = 0;

        // Uncomment the following block when the equivalent Rust implementation
        // for obtaining device properties is available.

        // let device_prop = tops_get_device_properties(0);
        // let arch_name = device_prop.gcu_arch_name;
        // if arch_name == "dtu-enflame-tops--gcu300" {
        if true {
            l1_mem = 1536 * 1024 - 1024;
            sip_cnt = 12;
            va_mem = 4096 * 16 * 2 /*VPT Thread*/ * 4;
        }

        let l31_m_num = ceil_div(info.M, UNIT_SIP_M);
        let l31_n_num = ceil_div(info.N, UNIT_SIP_N);

        let mut batch_multicore = false;
        let mut lhs_multicore = false;
        let mut rhs_multicore = false;

        if info.batch >= l31_m_num && info.batch >= l31_n_num {
            batch_multicore = true;
        } else if l31_m_num >= l31_n_num {
            lhs_multicore = true;
        } else {
            rhs_multicore = true;
        }

        let K_Align = align_up(info.K, UNIT_SIP_K);
        let N_Align = align_up(info.N, UNIT_SIP_N);

        if sum_mem(UNIT_SIP_M, UNIT_SIP_N, K_Align, BPE, N_Align) <= l1_mem {
            tune.sip_k = K_Align;
            if lhs_multicore || batch_multicore {
                tune.sip_m = UNIT_SIP_M;
                tune.sip_n = align_up(info.N, UNIT_SIP_N);

                if sum_mem(tune.sip_m, tune.sip_n, tune.sip_k, BPE, N_Align) > l1_mem
                    || sum_va_mem(tune.sip_m, tune.sip_n, 4) > va_mem
                {
                    let mut l1_mem_sip_n = (l1_mem / 2 / BPE - 2 * K_Align * tune.sip_m - N_Align)
                        / (2 * K_Align + tune.sip_m);
                    l1_mem_sip_n = align_down(l1_mem_sip_n, UNIT_SIP_N);
                    let mut va_mem_sip_n = va_mem / 16 / tune.sip_m;
                    va_mem_sip_n = align_down(va_mem_sip_n, UNIT_SIP_N);
                    tune.sip_n = l1_mem_sip_n;
                    if l1_mem_sip_n > va_mem_sip_n {
                        tune.sip_n = va_mem_sip_n;
                    }
                }

                if lhs_multicore {
                    set_split_option!(tune, false, true, false, false, false, false, false, false);
                } else if batch_multicore {
                    set_split_option!(tune, true, false, false, false, false, false, false, false);
                }
            } else if rhs_multicore {
                tune.sip_n = UNIT_SIP_N;
                tune.sip_m = align_up(info.M, UNIT_SIP_M);
                if sum_mem(tune.sip_m, tune.sip_n, tune.sip_k, BPE, N_Align) > l1_mem
                    || sum_va_mem(tune.sip_m, tune.sip_n, 4) > va_mem
                {
                    let mut l1_mem_sip_m = (l1_mem / 2 / BPE - 2 * K_Align * tune.sip_n - N_Align)
                        / (2 * K_Align + tune.sip_n);
                    l1_mem_sip_m = align_down(l1_mem_sip_m, UNIT_SIP_M);
                    let mut va_mem_sip_m = va_mem / 16 / tune.sip_n;
                    va_mem_sip_m = align_down(va_mem_sip_m, UNIT_SIP_M);
                    tune.sip_m = l1_mem_sip_m;
                    if l1_mem_sip_m > va_mem_sip_m {
                        tune.sip_m = va_mem_sip_m;
                    }
                }
                set_split_option!(tune, false, false, true, false, false, false, false, false);
            }
        } else {
            tune.sip_n = if info.N / UNIT_SIP_N > 1000 {
                UNIT_SIP_N * 4
            } else {
                UNIT_SIP_N
            };
            tune.sip_m = UNIT_SIP_M;
            tune.sip_k = if info.N / UNIT_SIP_N > 1000 {
                UNIT_SIP_K
            } else {
                K_Align
            };
            let l1_mem_sip_k = (l1_mem / 2 / BPE - tune.sip_m * tune.sip_n - N_Align)
                / (2 * tune.sip_m + 2 * tune.sip_n);
            if l1_mem_sip_k > 0 {
                tune.sip_k = align_down(l1_mem_sip_k, UNIT_SIP_K);
            }

            set_split_option!(
                tune,
                batch_multicore,
                lhs_multicore,
                rhs_multicore,
                false,
                false,
                false,
                false,
                false
            );
        }
        0
    }

    fn tuner_hgemm_half(&self, info: &AtenGemmInfo, tune: &mut AtenGemmTune) -> i32 {
        init_split!(info, tune);
        let sum_mem = |m: i64, n: i64, k: i64, bpe: i64, bias: i64| -> i64 {
            (2 * m * k + m * n + 2 * n * k + bias) * bpe * 2
        };

        let sum_va_mem = |m: i64, n: i64, bpe: i64| -> i64 { (m * n * 2) * bpe };

        let UNIT_SIP_M = if info.M * info.batch > 32 {
            64
        } else {
            if (info.M * info.batch <= 8) {
                1
            } else {
                32
            }
        };
        const UNIT_SIP_N: i64 = 128;
        let UNIT_SIP_K = if info.weight_type != DATATYPE::DataI4 {
            if info.M > 32 {
                64
            } else {
                128
            }
        } else {
            if info.M > 32 {
                128
            } else {
                256
            }
        };
        const BPE: i64 = 2;
        // let RBPE = if (info.weight_type == DATATYPE::DataBf16 || info.weight_type == DATATYPE::DataFp16) { 2 } else { 1 };
        let sip_cnt = 12; //todo!()
                          // const VDMEM_VALID_SIZE: i64 = 0x180000 - 0x8000 - 0x800;
        let l1_mem = 1536 * 1024 - 1024;
        let va_mem = 4096 * 16 * 2 * 4;

        let l31_m_num = ceil_div(info.M, UNIT_SIP_M);
        let l31_n_num = ceil_div(info.N, UNIT_SIP_N);

        let (batch_multicore, lhs_multicore, rhs_multicore) =
            if info.batch >= l31_m_num && info.batch >= l31_n_num {
                (true, false, false)
            } else if l31_m_num >= l31_n_num {
                (false, true, false)
            } else {
                (false, false, true)
            };

        let k_align = align_up(info.K, UNIT_SIP_K);
        let n_align = align_up(info.N, UNIT_SIP_N);

        if sum_mem(UNIT_SIP_M, UNIT_SIP_N, k_align, BPE, n_align) <= l1_mem {
            tune.sip_k = k_align;
            if lhs_multicore || batch_multicore {
                tune.sip_m = UNIT_SIP_M;
                tune.sip_n = align_up(info.N, UNIT_SIP_N);
                if sum_mem(tune.sip_m, tune.sip_n, tune.sip_k, BPE, n_align) > l1_mem
                    || sum_va_mem(tune.sip_m, tune.sip_n, 4) > va_mem
                {
                    let mut l1_mem_sip_n = (l1_mem / 2 / BPE - 2 * k_align * tune.sip_m - n_align)
                        / (2 * k_align + tune.sip_m);
                    l1_mem_sip_n = align_down(l1_mem_sip_n, UNIT_SIP_N);
                    let mut va_mem_sip_n = va_mem / 8 / tune.sip_m;
                    va_mem_sip_n = align_down(va_mem_sip_n, UNIT_SIP_N);
                    tune.sip_n = l1_mem_sip_n;
                    if l1_mem_sip_n > va_mem_sip_n {
                        tune.sip_n = va_mem_sip_n;
                    }
                }

                if lhs_multicore {
                    set_split_option!(tune, false, true, false, false, false, false, false, false);
                } else {
                    set_split_option!(tune, true, false, false, false, false, false, false, false);
                }
            } else if rhs_multicore {
                tune.sip_n = UNIT_SIP_N;
                tune.sip_m = align_up(info.M, UNIT_SIP_M);
                if sum_mem(tune.sip_m, tune.sip_n, tune.sip_k, BPE, n_align) > l1_mem
                    || sum_va_mem(tune.sip_m, tune.sip_n, 4) > va_mem
                {
                    let mut l1_mem_sip_m = (l1_mem / 2 / BPE - 2 * k_align * tune.sip_n - n_align)
                        / (2 * k_align + tune.sip_n);
                    l1_mem_sip_m = align_down(l1_mem_sip_m, UNIT_SIP_M);
                    let mut va_mem_sip_m = va_mem / 8 / tune.sip_n;
                    va_mem_sip_m = align_down(va_mem_sip_m, UNIT_SIP_M);
                    tune.sip_m = l1_mem_sip_m;
                    if l1_mem_sip_m > va_mem_sip_m {
                        tune.sip_m = va_mem_sip_m;
                    }
                }
                set_split_option!(tune, false, false, true, false, false, false, false, false);
            }
        } else {
            tune.sip_n = UNIT_SIP_N;
            tune.sip_m = UNIT_SIP_M;
            tune.sip_k = k_align;
            let l1_mem_sip_k = (l1_mem / 2 / BPE - tune.sip_m * tune.sip_n - n_align)
                / (2 * tune.sip_m + 2 * tune.sip_n);
            tune.sip_k = align_down(l1_mem_sip_k, UNIT_SIP_K);
            if tune.sip_k > info.K {
                tune.sip_k = info.K;
            }
            set_split_option!(tune, false, false, true, false, false, false, false, false);
        }
        if info.weight_type == DATATYPE::DataI4 && tune.sip_m > 128 && tune.sip_m % 128 != 0 {
            tune.sip_m = align_down(tune.sip_m, 128); //corner case for 4bit matmul
        }
        if tune.sip_m % 32 != 0 && tune.sip_m != 1 {
            //sip_m does not support, change to default
            tune.sip_m = 32;
        }
        // println!("TunerHGemmF16");
        0
    }
}
