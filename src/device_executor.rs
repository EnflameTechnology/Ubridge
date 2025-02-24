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
* @file    device_executor.rs
* @brief
*
* @author  Guoqing Bao
* @date    2022-10-27 - 2025-01-03
* @version V0.1
* @par     Copyright (c) Enflame Tech Company.
* @par     History: Add supports for fused kernels
* @par     Comments: a gcu executor for kernel management, and gcu kernel computations.
*/
use core::fmt::Debug;
use core::panic;
use std::collections::HashMap;
use std::fs;
use tops::driv;

//Import UHAL for common computing interfaces
use uhal::error::DeviceResult;
use uhal::function::FunctionTrait;
use uhal::module::ModuleTrait;
//Tops backend

use tops::module::TopsModule as Module;
use tops_backend as tops;

#[derive(Debug)]
pub struct ModuleX(Module);

#[derive(Debug)]
pub struct FuncX(driv::topsFunction_t);
impl FuncX {
    pub fn as_inner(&self) -> driv::topsFunction_t {
        self.0
    }
}

unsafe impl Send for ModuleX {}
unsafe impl Sync for ModuleX {}
unsafe impl Send for FuncX {}
unsafe impl Sync for FuncX {}

#[derive(Debug)]
pub struct DeviceExecutor {
    pub module_map: HashMap<String, ModuleX>,
    pub function_map: HashMap<String, FuncX>,
}

impl DeviceExecutor {
    pub fn init_kernels(&mut self, kernel_platform: &str) -> DeviceResult<()> {
        let full_kernel_folder =
            format!("{}/kernels/{}", env!("CARGO_MANIFEST_DIR"), kernel_platform).to_string();
        let paths = fs::read_dir(&full_kernel_folder).unwrap();
        for path in paths {
            let p = path.unwrap().path();
            let file = p.file_name().unwrap();
            let filename = file.to_str().unwrap();
            let kernel_name = p.file_stem().unwrap().to_str().unwrap();
            if filename.ends_with(".topsfb") || filename.ends_with(".ptx") {
                #[cfg(feature = "cuda_backend")]
                let ptx = format!("{}/{}.ptx", full_kernel_folder, kernel_name).to_string();

                #[cfg(feature = "tops_backend")]
                let ptx = format!("{}/{}.topsfb", full_kernel_folder, kernel_name).to_string();

                println!("{}", ptx);
                let module = ModuleX {
                    0: Module::from_file(&ptx).unwrap(),
                };
                self.module_map.insert(kernel_name.to_string(), module);
            }
        }

        if self.module_map.len() > 0 {
            println!("{} kernel(s) loaded!", self.module_map.len());
        }
        Ok(())
    }

    pub fn new(device_id: u32) -> Self {
        println!("DeviceExecutor::new");

        #[cfg(feature = "tops_backend")]
        let kernel_platform = "scorpio"; //default kernel path

        #[cfg(feature = "scorpio")]
        let kernel_platform = "scorpio";

        let unary_functions = vec![
            "uneg",
            "uexp",
            "ulog",
            "usin",
            "ucos",
            "uabs",
            "usqr",
            "usqrt",
            "ursqrt",
            "ugelu",
            "ugelu_erf",
            "urelu",
            "utanh",
            "urecip",
            "uelu",
            "usigmoid",
            "usilu",
            "uelu",
        ];

        let binary_functions = vec![
            "badd", "bsub", "bmul", "bdiv", "bmaximum", "bminimum", "mod", "eq", "ne", "ge", "gt",
            "lt", "le",
        ];

        let cast_functions = vec![
            "cast_f16_i16",
            "cast_f16_i32",
            "cast_f16_f32",
            "cast_f32_i16",
            "cast_f32_i32",
            "cast_f32_f16",
            // "cast_i8_i16", "cast_i8_i32", "cast_i8_f32", "cast_i8_f16", "cast_f16_i8", "cast_f32_i8", "cast_i16_i8", "cast_i32_i8",
            "cast_bf16_i16",
            "cast_bf16_i32",
            "cast_bf16_f16",
            "cast_bf16_f32",
            "cast_f16_bf16",
            "cast_f32_bf16",
            "cast_i16_bf16",
            "cast_i32_bf16",
            "cast_u8_bf16",
            "cast_u16_bf16",
            "cast_u32_bf16",
            "cast_i16_i32",
            "cast_i16_f32",
            "cast_i16_f16",
            "cast_i32_i16",
            "cast_i32_f32",
            "cast_i32_f16",
            "cast_u8_u16",
            "cast_u8_u32",
            "cast_u8_f32",
            "cast_u8_f16",
            "cast_u16_u8",
            "cast_u16_u32",
            "cast_u16_f32",
            "cast_u16_f16",
            "cast_u32_u8",
            "cast_u32_u16",
            "cast_u32_f32",
            "cast_u32_f16",
        ];

        let reduce_functions = vec![
            "fast_min_f32",
            "fast_min_f16",
            "fast_min_i8",
            "fast_min_bf16",
            "fast_max_f32",
            "fast_max_f16",
            "fast_max_i8",
            "fast_max_bf16",
            "fast_sum_f32",
            "fast_sum_f16",
            "fast_sum_i8",
            "fast_sum_bf16",
            "fast_argmin_f32",
            "fast_argmin_f16",
            "fast_argmin_i8",
            "fast_argmin_bf16",
            "fast_argmax_f32",
            "fast_argmax_f16",
            "fast_argmax_i8",
            "fast_argmax_bf16",
            "softmax_f16",
            "softmax_bf16",
            "softmax_f32",
            "layernorm_f16",
            "layernorm_bf16",
            "layernorm_f32",
        ];

        let where_functions = vec![
            "where_i64_f32",
            "where_i64_f64",
            "where_i64_u8",
            "where_i64_u32",
            "where_i64_i64",
            "where_u32_f32",
            "where_u32_f64",
            "where_u32_u8",
            "where_u32_u32",
            "where_u32_i64",
            "where_u8_f32",
            "where_u8_f64",
            "where_u8_u8",
            "where_u8_u32",
            "where_u8_i64",
            "where_u8_bf16",
            "where_u8_f16",
        ];

        let index_functions = vec![
            "is_i64_bf16",
            "is_u32_bf16",
            "is_u8_bf16",
            "is_i64_f16",
            "is_u32_f16",
            "is_u8_f16",
            "is_i64_f32",
            "is_i64_f64",
            "is_i64_u8",
            "is_i64_u32",
            "is_i64_i64",
            "is_u32_f32",
            "is_u32_f64",
            "is_u32_u8",
            "is_u32_i64",
            "is_u32_u32",
            "is_u8_f32",
            "is_u8_f64",
            "is_u8_u8",
            "is_u8_u32",
            "is_u8_i64",
        ];

        let copy_functions = vec![
            "ucopy_bf16",
            "ucopy_u8",
            "ucopy_i8",
            "ucopy_u32",
            "ucopy_f64",
            "ucopy_f16",
            "ucopy_f32",
        ];

        let kvconcat_functions = vec![
            "kvconcat_bf16",
            "kvconcat_u8",
            "kvconcat_f64",
            "kvconcat_f16",
            "kvconcat_f32",
        ];

        let embedding_functions = vec![
            "rope_f32",
            "rope_f16",
            "rope_bf16",
            "rope_f32_f16",
            "rope_f32_bf16",
        ];

        let conv_functions = vec![
            "conv1d_f32",
            "conv1d_f16",
            "conv1d_bf16",
            "conv1d_f64",
            "conv1d_u8",
            "conv1d_u32",
            "conv2d_f32",
            "conv2d_f16",
            "conv2d_bf16",
        ];

        let quant_functions = vec![
            "quantize_block_q8_0_f16",
            "quantize_block_q8_0_bf16",
            "quantize_block_q8_0_f32",
            "dequantize_block_q8_0_f16",
            "dequantize_block_q8_0_bf16",
            "dequantize_block_q8_0_f32",
            "quantize_block_q4_k_f16",
            "quantize_block_q4_k_bf16",
            "quantize_block_q4_k_f32",
            "dequantize_block_q4_k_f16",
            "dequantize_block_q4_k_bf16",
            "dequantize_block_q4_k_f32",
        ];

        let cache_functions = vec![
            "reshape_and_cache_f32",
            "reshape_and_cache_f16",
            "reshape_and_cache_bf16",
        ];
        let attention_functions = vec!["paged_attention_v1_f16", "paged_attention_v1_bf16"];
        let sort_functions = vec!["asort_asc", "asort_desc"];
        let gather_functions = vec![
            "gather_i64",
            "gather_u8",
            "gather_u32",
        ];

        //index_add
        let ia_functions = vec![
            "ia_i64",
            "ia_u8",
            "ia_u32",
        ];

        let mut executor = DeviceExecutor {
            module_map: HashMap::<String, ModuleX>::new(),
            function_map: HashMap::<String, FuncX>::new(),
        };
        match executor.init_kernels(kernel_platform) {
            Ok(()) => {
                for module in executor.module_map.keys().by_ref() {
                    match module.as_str() {
                        "unary" => {
                            for dt in ["bf16", "f16", "f32"] {
                                for func in &unary_functions {
                                    let name = format!("{}_{}", func, dt);
                                    println!("Load function {}", name);
                                    let function = executor.get_function(module, &name);
                                    executor.function_map.insert(name, function);
                                }
                            }
                        }
                        "binary" => {
                            for dt in ["bf16", "f16", "f32", "u32"] {
                                for func in &binary_functions {
                                    let name = format!("{}_{}", func, dt);
                                    println!("Load function {}", name);
                                    let function = executor.get_function(module, &name);
                                    executor.function_map.insert(name, function.into());
                                }
                            }
                        }
                        "fill" => {
                            for dt in ["bf16", "f16", "f32", "f64", "i32", "i16", "i8", "bool"] {
                                let name = format!("{}_{}", module, dt);
                                println!("Load function {}", name);
                                let function = executor.get_function(module, &name);
                                executor.function_map.insert(name, function.into());
                            }
                        }
                        "matmul" => {
                            for dt in [
                                "bf16",
                                "f16",
                                "f32",
                                "f16_4bit",
                                "f16_8bit",
                                "bf16_4bit",
                                "bf16_8bit",
                            ] {
                                let name = format!("{}_{}", module, dt);
                                println!("Load function {}", name);
                                let function = executor.get_function(module, &name);
                                executor.function_map.insert(name, function.into());
                            }
                        }
                        "affine" => {
                            for dt in ["bf16", "f16", "f32"] {
                                let name = format!("{}_{}", module, dt);
                                println!("Load function {}", name);
                                let function = executor.get_function(module, &name);
                                executor.function_map.insert(name, function.into());
                            }
                        }
                        "cast" => {
                            for func in &cast_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "reduce" => {
                            for func in &reduce_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "ternary" => {
                            for func in &where_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "indexing" => {
                            for func in &index_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }

                            for func in &gather_functions {
                                for dt in ["bf16", "f16", "f32", "f64", "u8", "u32", "i64"] {
                                    let name = format!("{}_{}", func, dt);
                                    println!("Load function {}", name);
                                    let function = executor.get_function(module, &name);
                                    executor.function_map.insert(name, function.into());
                                }
                            }

                            for func in &ia_functions {
                                for dt in ["bf16", "f16", "f32", "f64", "u8", "u32", "i64"] {
                                    let name = format!("{}_{}", func, dt);
                                    println!("Load function {}", name);
                                    let function = executor.get_function(module, &name);
                                    executor.function_map.insert(name, function.into());
                                }
                            }
                        }
                        "embedding" => {
                            for func in &embedding_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "kvconcat" => {
                            for func in &kvconcat_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "conv" => {
                            for func in &conv_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "copy" => {
                            for func in &copy_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "quant" => {
                            for func in &quant_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "cache" => {
                            for func in &cache_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "attention" => {
                            for func in &attention_functions {
                                println!("Load function {}", func);
                                let function = executor.get_function(module, &func.to_string());
                                executor
                                    .function_map
                                    .insert(func.to_string(), function.into());
                            }
                        }
                        "sort" => {
                            for dt in ["bf16", "f16", "f32", "f64", "u8", "u32", "i64"] {
                                for func in &sort_functions {
                                    let name = format!("{}_{}", func, dt);
                                    println!("Load function {}", name);
                                    let function = executor.get_function(module, &name);
                                    executor.function_map.insert(name, function.into());
                                }
                            }
                        }
                        _ => {
                            println!("Module not load: {}", module);
                        }
                    }
                }
            }
            _ => panic!("Load kernels failed!"),
        };
        executor
    }

    pub fn get_function(&self, module: &String, name: &String) -> FuncX {
        FuncX {
            0: self.module_map[module]
                .0
                .get_function(name)
                .unwrap()
                .to_raw(),
        }
    }

    pub fn has_function(&self, module_name: String, func_name: String) -> bool {
        if self.module_map.contains_key(&module_name) {
            return self.function_map.contains_key(&func_name);
        }
        false
    }

    // pub fn synchronize(&self) -> DeviceResult<()> {
    //     match &self.stream {
    //         Some(stream) => stream.synchronize(),
    //         _ => {
    //             panic!("Invalid stream!")
    //         }
    //     }
    // }
}
//     pub fn unary_compute_owned(
//         &self,
//         op: DeviceOpCode,
//         arg: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<()> {
//         match op {
//             DeviceOpCode::RELU => self.activation_inplace(arg, eager_mode, "relu".to_string()),
//             DeviceOpCode::GELU => self.activation_inplace(arg, eager_mode, "gelu".to_string()),
//             DeviceOpCode::LEAKY => self.activation_inplace(arg, eager_mode, "leaky".to_string()),
//             DeviceOpCode::TANH => self.activation_inplace(arg, eager_mode, "tanh".to_string()),
//             // DeviceOpCode::Transpose => self.transpose_inpace(arg, eager_mode),
//             _ => panic!("Not supported operation!"),
//         }
//     }

//     pub fn binary_compute_owned(
//         &self,
//         op: DeviceOpCode,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         match op {
//             DeviceOpCode::AddF => self.addf32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::SubF => self.subf32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::MulF => self.mulf32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::DivF => self.divf32_owned(lhs, rhs, eager_mode),

//             DeviceOpCode::AddI => self.addi32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::SubI => self.subi32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::MulI => self.muli32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::DivI => self.divi32_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::MatMulF => self.matmul_owned(lhs, rhs, eager_mode),
//             DeviceOpCode::Conv2DF => self.conv2d_owned(lhs, rhs, eager_mode),
//             _ => panic!("Not supported operation!"),
//         }
//     }

//     pub fn mock_result(
//         &self,
//         mock_data: Vec<f32>,
//         mock_shape: Vec<usize>,
//     ) -> DeviceResult<DeviceTensor> {
//         let data = DeviceBuffer::from_slice(&mock_data);
//         match data {
//             Ok(buf) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(buf)),
//                 shape: mock_shape,
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }
//     }

//     pub fn addf32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
//         self.elementf32_owned(lhs, rhs, 0i32, eager_mode)
//     }

//     pub fn subf32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![0.0f32; 6], vec![2, 3])
//         self.elementf32_owned(lhs, rhs, 1i32, eager_mode)
//     }

//     pub fn mulf32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
//         self.elementf32_owned(lhs, rhs, 2i32, eager_mode)
//     }

//     pub fn divf32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![1.0f32; 6], vec![2, 3])
//         self.elementf32_owned(lhs, rhs, 3i32, eager_mode)
//     }

//     pub fn addi32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
//         self.elementi32_owned(lhs, rhs, 0i32, eager_mode)
//     }

//     pub fn subi32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![0.0f32; 6], vec![2, 3])
//         self.elementi32_owned(lhs, rhs, 1i32, eager_mode)
//     }

//     pub fn muli32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
//         self.elementi32_owned(lhs, rhs, 2i32, eager_mode)
//     }

//     pub fn divi32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         // self.mock_result(vec![1.0f32; 6], vec![2, 3])
//         self.elementi32_owned(lhs, rhs, 3i32, eager_mode)
//     }

//     pub fn get_block_grid(&self, shape1: usize, shape0: usize) -> (usize, usize, usize) {
//         let grid_a: usize = (shape1 + 16 - 1) / 16;
//         let grid_b: usize = (shape0 + 16 - 1) / 16;
//         (16, grid_a, grid_b)
//     }

//     #[allow(non_snake_case)]
//     pub fn elementf32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         tp: i32,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         let kernel = if self.function_map.contains_key("elementf32") {
//             Function::new(
//                 self.function_map["elementf32"].0,
//                 &self.module_map["unary"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         let size: usize = lhs.shape.iter().product();
//         let matOut = DeviceBuffer::from_slice(&vec![0.0f32; size])?;
//         let (_block_size, _grid_a, _grid_b) = self.get_block_grid(
//             if rhs.shape.len() > 1 {
//                 rhs.shape[1]
//             } else {
//                 lhs.shape[0]
//             },
//             lhs.shape[0],
//         );

//         let result: DeviceResult<()> = match (&lhs.data, &rhs.data, &self.stream) {
//             (Some(data_left), Some(data_right), Some(stream)) => match (data_left, data_right) {
//                 (DeviceTensorKind::FloatTensor(matA), DeviceTensorKind::FloatTensor(matB)) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         size as i32,
//                         tp
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         lhs.shape[0] as u32,
//                         if lhs.shape.len() > 1 {lhs.shape[1] as u32 } else {lhs.shape[0] as u32},
//                         tp as u32
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {
//                         println!("Stream synchronized!");
//                     }
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(matOut)),
//                 shape: lhs.shape.clone(),
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }
//     }

//     #[allow(non_snake_case)]
//     pub fn elementi32_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         tp: i32,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         let kernel = if self.function_map.contains_key("elementi32") {
//             Function::new(
//                 self.function_map["elementi32"].0,
//                 &self.module_map["unary"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         let size: usize = lhs.shape.iter().product();
//         let matOut = DeviceBuffer::from_slice(&vec![0i32; size])?;

//         #[cfg(feature = "cuda_backend")]
//         let (block_size, grid_a, grid_b) = self.get_block_grid(
//             if rhs.shape.len() > 1 {
//                 rhs.shape[1]
//             } else {
//                 lhs.shape[0]
//             },
//             lhs.shape[0],
//         );

//         let result: DeviceResult<()> = match (&lhs.data, &rhs.data, &self.stream) {
//             (Some(data_left), Some(data_right), Some(stream)) => match (data_left, data_right) {
//                 (DeviceTensorKind::Int32Tensor(matA), DeviceTensorKind::Int32Tensor(matB)) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         size as i32,
//                         tp
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         lhs.shape[0] as u32,
//                         if lhs.shape.len() > 1 {lhs.shape[1] as u32 } else {lhs.shape[0] as u32},
//                         tp as u32
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {
//                         println!("Stream synchronized!");
//                     }
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(matOut)),
//                 shape: lhs.shape.clone(),
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }
//     }

//     //Maximum input size 512 x 512 supported!
//     #[allow(non_snake_case)]
//     pub fn matmul_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         let kernel = if self.function_map.contains_key("matmul") {
//             Function::new(self.function_map["matmul"].0, &self.module_map["matmul"].0)
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         #[cfg(feature = "tops_backend")]
//         let inputShapeA =
//             DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
//         #[cfg(feature = "tops_backend")]
//         let inputShapeB =
//             DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1] as i32, 1i32, 1i32])?;

//         let matOut = DeviceBuffer::from_slice(&vec![0.0f32; lhs.shape[0] * rhs.shape[1]])?;
//         #[cfg(feature = "cuda_backend")]
//         let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

//         println!("GCU: Left {:?}, Right {:?}", lhs.shape, rhs.shape);

//         let result: DeviceResult<()> = match (&lhs.data, &rhs.data, &self.stream) {
//             (Some(data_left), Some(data_right), Some(stream)) => match (data_left, data_right) {
//                 (DeviceTensorKind::FloatTensor(matA), DeviceTensorKind::FloatTensor(matB)) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         inputShapeA.as_device_ptr(),
//                         inputShapeB.as_device_ptr()
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         lhs.shape[0] as u32,
//                         lhs.shape[1] as u32,
//                         rhs.shape[1] as u32
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {}
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(matOut)),
//                 shape: vec![lhs.shape[0], rhs.shape[1]],
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }

//         // self.mock_result(vec![23.0f32; 17 * 18], vec![17, 18])
//     }

//     #[allow(non_snake_case)]
//     pub fn batch_matmul(
//         &mut self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         out: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<()> {
//         let kernel = if self.function_map.contains_key("batch_matmul") {
//             Function::new(
//                 self.function_map["batch_matmul"].0,
//                 &self.module_map["matmul"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         // #[cfg(feature = "tops_backend")]
//         // let inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, lhs.shape[2] as i32])?;
//         // #[cfg(feature = "tops_backend")]
//         // let inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1]  as i32, rhs.shape[2]  as i32])?;
//         let shape1 = format!(
//             "inputShapeA{}_{}_{}",
//             lhs.shape[0], lhs.shape[1], lhs.shape[2]
//         );
//         if !self.cache_shape.contains_key(&shape1) {
//             let buffer = Box::new(DeviceBuffer::from_slice(&[
//                 lhs.shape[0] as i32,
//                 lhs.shape[1] as i32,
//                 lhs.shape[2] as i32,
//             ])?);
//             self.cache_shape.insert(shape1.clone(), buffer);
//         }

//         let shape2 = format!(
//             "inputShapeB{}_{}_{}",
//             rhs.shape[0], rhs.shape[1], rhs.shape[2]
//         );
//         if !self.cache_shape.contains_key(&shape2) {
//             let buffer = Box::new(DeviceBuffer::from_slice(&[
//                 rhs.shape[0] as i32,
//                 rhs.shape[1] as i32,
//                 rhs.shape[2] as i32,
//             ])?);
//             self.cache_shape.insert(shape2.clone(), buffer);
//         }
//         let inputShapeA = &self.cache_shape[&shape1];
//         let inputShapeB = &self.cache_shape[&shape2];

//         let cachename1 = format!(
//             "matTranpose{}_{}_{}",
//             rhs.shape[0], rhs.shape[1], rhs.shape[2]
//         );

//         if !self.cache_buffer.contains_key(&cachename1) {
//             let buffer = Box::new(
//                 DeviceTensor::from_vec_shape(
//                     &vec![0.0f32; rhs.shape[0] * rhs.shape[1] * rhs.shape[2]],
//                     vec![rhs.shape[0], rhs.shape[1], rhs.shape[2]],
//                 )
//                 .unwrap(),
//             );
//             self.cache_buffer.insert(cachename1.clone(), buffer);
//             println!("GCU cache buffer [{}, {}]", rhs.shape[1], rhs.shape[2]);
//         }

//         // let cachename = format!("matOut{}_{}_{}", lhs.shape[0], lhs.shape[1], rhs.shape[2]);

//         // if !self.cache_buffer.contains_key(&cachename) {
//         //     let buffer = Box::new(
//         //         DeviceTensor::from_vec_shape(
//         //             &vec![0.0f32; lhs.shape[0] * lhs.shape[1] * rhs.shape[2]],
//         //             vec![lhs.shape[0], lhs.shape[1], rhs.shape[2]],
//         //         )
//         //         .unwrap(),
//         //     );
//         //     self.cache_buffer.insert(cachename.clone(), buffer);
//         //     println!("GCU cache buffer [{}, {}]", lhs.shape[1], rhs.shape[2]);
//         // }

//         let matTranpose = &self.cache_buffer[&cachename1];
//         // let matOut = &self.cache_buffer[&cachename];

//         let result: DeviceResult<()> = match (
//             &lhs.data,
//             &rhs.data,
//             &self.stream,
//             &matTranpose.data,
//             &out.data,
//         ) {
//             (
//                 Some(data_left),
//                 Some(data_right),
//                 Some(stream),
//                 Some(data_transpose),
//                 Some(data_out),
//             ) => match (data_left, data_right, data_transpose, data_out) {
//                 (
//                     DeviceTensorKind::FloatTensor(matA),
//                     DeviceTensorKind::FloatTensor(matB),
//                     DeviceTensorKind::FloatTensor(matTrans),
//                     DeviceTensorKind::FloatTensor(matO),
//                 ) => unsafe {
//                     let batch = lhs.shape[0] as u32;
//                     let W = lhs.shape[1] as u32;

//                     println!(
//                         "GCU: Left {:?}, Right {:?} [{}, {}]",
//                         lhs.shape, rhs.shape, batch, W
//                     );

//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(W, batch, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matTrans.as_device_ptr(),
//                         matO.as_device_ptr(),
//                         inputShapeA.as_device_ptr(),
//                         inputShapeB.as_device_ptr()
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {}
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(()),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }
//     }

//     #[allow(non_snake_case)]
//     // pub fn transposed_matmul_owned(
//     //     &mut self,
//     //     lhs: &DeviceTensor,
//     //     rhs: &DeviceTensor,
//     //     eager_mode: bool,
//     // ) -> DeviceResult<&DeviceTensor> {
//     //     let cachename = format!("matOut{}_{}_{}", lhs.shape[0], lhs.shape[1], rhs.shape[2]);

//     //     if !self.cache_buffer.contains_key(&cachename) {
//     //         let buffer = Box::new(
//     //             DeviceTensor::from_vec_shape(
//     //                 &vec![0.0f32; lhs.shape[0] * lhs.shape[1] * rhs.shape[2]],
//     //                 vec![lhs.shape[0], lhs.shape[1], rhs.shape[2]],
//     //             )
//     //             .unwrap(),
//     //         );
//     //         self.cache_buffer.insert(cachename.clone(), buffer);
//     //         println!("GCU cache buffer [{}, {}]", lhs.shape[1], rhs.shape[2]);
//     //     }
//     //     // let matOut = &self.cache_buffer[&cachename];

//     //     match self.transposed_matmul(lhs, rhs, &self.cache_buffer[&cachename], eager_mode) {
//     //         Ok(_) => {
//     //             Ok(&self.cache_buffer[&cachename])
//     //         }
//     //         _=> {
//     //             panic!("Unable to use kernel!");
//     //         }
//     //     }
//     // }

//     // pub fn matmul_f32(&mut self, shape: &[usize], lhs: &topsDeviceptr_t, rhs: &topsDeviceptr_t, out: &topsDeviceptr_t, eager_mode: bool) -> DeviceResult<()> {

//     //     let (b, m, n, k) = (shape[0], shape[1], shape[2], shape[3]);
//     //     let ltensor = DeviceTensor::from_raw_parts(lhs, b * m * n, vec![b, m, n]).unwrap();
//     //     let rtensor = DeviceTensor::from_raw_parts(rhs, b * n * k, vec![b, n, k]).unwrap();
//     //     let outensor = DeviceTensor::from_raw_parts(out, b * m * k, vec![b, m, k]).unwrap();

//     //     if k > 1 && m < 12000 && k < 12000 && n < 12000 {
//     //         if b > 1 {
//     //             return self.batch_matmul(&ltensor, &rtensor, &outensor, eager_mode);
//     //         } else {
//     //             return self.transposed_matmul(&ltensor, &rtensor, &outensor, eager_mode);
//     //         }

//     //     }
//     //     Ok(())

//     // }
//     #[allow(non_snake_case)]
//     pub fn transposed_matmul(
//         &mut self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         out: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<()> {
//         let kernel_transpose = if self.function_map.contains_key("transpose_kernel") {
//             Function::new(
//                 self.function_map["transpose_kernel"].0,
//                 &self.module_map["transpose"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         let kernel_matmul = if self.function_map.contains_key("transposed_matmul") {
//             Function::new(
//                 self.function_map["transposed_matmul"].0,
//                 &self.module_map["transpose"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };

//         let shape1 = format!(
//             "inputShapeA{}_{}_{}",
//             lhs.shape[0], lhs.shape[1], lhs.shape[2]
//         );
//         if !self.cache_shape.contains_key(&shape1) {
//             let buffer = Box::new(DeviceBuffer::from_slice(&[
//                 lhs.shape[0] as i32,
//                 lhs.shape[1] as i32,
//                 lhs.shape[2] as i32,
//             ])?);
//             self.cache_shape.insert(shape1.clone(), buffer);
//         }

//         let shape2 = format!(
//             "inputShapeB{}_{}_{}",
//             rhs.shape[0], rhs.shape[1], rhs.shape[2]
//         );
//         if !self.cache_shape.contains_key(&shape2) {
//             let buffer = Box::new(DeviceBuffer::from_slice(&[
//                 rhs.shape[0] as i32,
//                 rhs.shape[1] as i32,
//                 rhs.shape[2] as i32,
//             ])?);
//             self.cache_shape.insert(shape2.clone(), buffer);
//         }

//         let shape3 = format!(
//             "transposedShapeB{}_{}_{}",
//             rhs.shape[0], rhs.shape[2], rhs.shape[1]
//         );
//         if !self.cache_shape.contains_key(&shape3) {
//             let buffer = Box::new(DeviceBuffer::from_slice(&[
//                 rhs.shape[0] as i32,
//                 rhs.shape[2] as i32,
//                 rhs.shape[1] as i32,
//             ])?);
//             self.cache_shape.insert(shape3.clone(), buffer);
//         }
//         // #[cfg(feature = "tops_backend")]
//         // let inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, lhs.shape[2] as i32])?;
//         // #[cfg(feature = "tops_backend")]
//         // let inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1]  as i32, rhs.shape[2]  as i32])?;

//         // let transposedShapeB = DeviceBuffer::from_slice(&[rhs.shape[0]  as i32, rhs.shape[2]  as i32, rhs.shape[1]  as i32])?;

//         let inputShapeA = &self.cache_shape[&shape1];
//         let inputShapeB = &self.cache_shape[&shape2];
//         let transposedShapeB = &self.cache_shape[&shape3];

//         let cachename1 = format!(
//             "matTranpose{}_{}_{}",
//             rhs.shape[0], rhs.shape[1], rhs.shape[2]
//         );

//         if !self.cache_buffer.contains_key(&cachename1) {
//             let buffer = Box::new(
//                 DeviceTensor::from_vec_shape(
//                     &vec![0.0f32; rhs.shape[0] * rhs.shape[1] * rhs.shape[2]],
//                     vec![rhs.shape[0], rhs.shape[1], rhs.shape[2]],
//                 )
//                 .unwrap(),
//             );
//             self.cache_buffer.insert(cachename1.clone(), buffer);
//             println!("GCU cache buffer [{}, {}]", rhs.shape[1], rhs.shape[2]);
//         }

//         let matTranpose = &self.cache_buffer[&cachename1];

//         let result: DeviceResult<()> = match (
//             &lhs.data,
//             &rhs.data,
//             &self.stream,
//             &matTranpose.data,
//             &out.data,
//         ) {
//             (
//                 Some(data_left),
//                 Some(data_right),
//                 Some(stream),
//                 Some(data_transpose),
//                 Some(data_out),
//             ) => match (data_left, data_right, data_transpose, data_out) {
//                 (
//                     DeviceTensorKind::FloatTensor(matA),
//                     DeviceTensorKind::FloatTensor(matB),
//                     DeviceTensorKind::FloatTensor(matTrans),
//                     DeviceTensorKind::FloatTensor(matO),
//                 ) => unsafe {
//                     let N = rhs.shape[2] as u32;
//                     let M = rhs.shape[1] as u32;
//                     let TILE_DIM = 64;
//                     let mut GRIDS = N / TILE_DIM;
//                     if GRIDS * TILE_DIM < N {
//                         GRIDS += 1
//                     };
//                     let mut BLOCKS = M / TILE_DIM;
//                     if BLOCKS * TILE_DIM < M {
//                         BLOCKS += 1
//                     };
//                     let mut PER_BLOCKS = 1;
//                     if BLOCKS > 4 {
//                         PER_BLOCKS = 4;
//                         if (BLOCKS / PER_BLOCKS) * 4 < BLOCKS {
//                             BLOCKS /= PER_BLOCKS;
//                             BLOCKS += 1;
//                         } else {
//                             BLOCKS /= PER_BLOCKS;
//                         }
//                     }

//                     #[cfg(feature = "tops_backend")]
//                     let _result = launch!(kernel_transpose<<<(GRIDS, BLOCKS, 1), (PER_BLOCKS, 1, 1), 0, stream>>>(
//                         matB.as_device_ptr(),
//                         matTrans.as_device_ptr(),
//                         inputShapeB.as_device_ptr()
//                     ));

//                     let K = lhs.shape[1] as u32;
//                     let mut threads = 4;
//                     if K % 4 > 0 {
//                         threads += 1;
//                     }
//                     let mut grids = K / 4;
//                     if grids < 1 {
//                         threads = K;
//                         grids = 1;
//                     }
//                     #[cfg(feature = "tops_backend")]
//                     let result1 = launch!(kernel_matmul<<<(grids, 1, 1), (threads, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matTrans.as_device_ptr(),
//                         matO.as_device_ptr(),
//                         inputShapeA.as_device_ptr(),
//                         transposedShapeB.as_device_ptr()
//                     ));
//                     println!(
//                         "GCU: Left {:?}, Right {:?} Transpose [{}, {}], Dot [{}, {}]",
//                         lhs.shape,
//                         rhs.shape,
//                         GRIDS,
//                         BLOCKS * PER_BLOCKS,
//                         grids,
//                         threads
//                     );

//                     result1
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {}
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(ret) => Ok(ret),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }

//         // self.mock_result(vec![23.0f32; 17 * 18], vec![17, 18])
//     }

//     #[allow(non_snake_case)]
//     pub fn conv2d_owned(
//         &self,
//         lhs: &DeviceTensor,
//         rhs: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         let kernel = if self.function_map.contains_key("convolution") {
//             Function::new(
//                 self.function_map["convolution"].0,
//                 &self.module_map["conv"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         #[cfg(feature = "tops_backend")]
//         let inputShapeA =
//             DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
//         #[cfg(feature = "tops_backend")]
//         let inputShapeB =
//             DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1] as i32, 1i32, 1i32])?;
//         #[cfg(feature = "tops_backend")]
//         let channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

//         let matOut = DeviceBuffer::from_slice(&vec![
//             0.0f32;
//             (lhs.shape[0] - rhs.shape[0] + 1)
//                 * (lhs.shape[1] - rhs.shape[1] + 1)
//         ])?;

//         #[cfg(feature = "cuda_backend")]
//         let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

//         let result: DeviceResult<()> = match (&lhs.data, &rhs.data, &self.stream) {
//             (Some(data_left), Some(data_right), Some(stream)) => match (data_left, data_right) {
//                 (DeviceTensorKind::FloatTensor(matA), DeviceTensorKind::FloatTensor(matB)) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         inputShapeA.as_device_ptr(),
//                         inputShapeB.as_device_ptr(),
//                         channelInfo.as_device_ptr()
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matB.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         lhs.shape[0] as i32,
//                         lhs.shape[1] as i32,
//                         rhs.shape[0] as i32,
//                         rhs.shape[1] as i32
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {
//                         println!("Stream synchronized!");
//                     }
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(matOut)),
//                 shape: vec![
//                     (lhs.shape[0] - rhs.shape[0] + 1),
//                     (lhs.shape[1] - rhs.shape[1] + 1),
//                 ],
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }

//         // self.mock_result(vec![1.0f32; 6], vec![2, 3])
//     }

//     #[allow(non_snake_case)]
//     pub fn activation_inplace(
//         &self,
//         arg: &DeviceTensor,
//         eager_mode: bool,
//         act_type: String,
//     ) -> DeviceResult<()> {
//         let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);
//         if !["relu", "gelu", "leaky", "tanh"].contains(&act_type.as_str()) {
//             panic!("Activation type not supported!");
//         }

//         let kernel = if self.function_map.contains_key("activationf32") {
//             Function::new(
//                 self.function_map["activationf32"].0,
//                 &self.module_map["unary"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         let size: usize = arg.shape.iter().product();

//         #[cfg(feature = "cuda_backend")]
//         let (block_size, grid_a, grid_b) = self.get_block_grid(arg.shape[1], arg.shape[0]);

//         let result: DeviceResult<()> = match (&arg.data, &self.stream) {
//             (Some(data_left), Some(stream)) => match data_left {
//                 DeviceTensorKind::FloatTensor(matA) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         size as i32,
//                         map_act[act_type.as_str()]
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         arg.shape[0] as u32,
//                         arg.shape[1] as u32,
//                         map_act[act_type.as_str()] as i32
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {
//                         println!("Stream synchronized!");
//                     }
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }
//         result
//         // match result {
//         //     Ok(_) => {
//         //         // match &arg.data {
//         //         //     Some(data) => {
//         //         //         match data {
//         //         //             DeviceTensorKind::FloatTensor(matA) => {
//         //         //                 Ok(DeviceTensor {
//         //         //                     data: Some(DeviceTensorKind::from(matA)),
//         //         //                     shape: arg.shape,
//         //         //                 })
//         //         //             }
//         //         //             _ => { panic!("Data type not supported!"); }
//         //         //         }
//         //         //     }
//         //         //     _ => { panic!("Unable to return data!");}
//         //         // }
//         //         Ok(())

//         //     }
//         //     #[cfg(test)]
//         //     Err(_e) => { panic!("Failed to alloc device memory!"); }
//         //     #[cfg(not(test))]
//         //     Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
//         // }
//     }

//     //Maximum input size 512 x 512 supported!
//     #[allow(non_snake_case)]
//     pub fn transpose_owned(
//         &self,
//         arg: &DeviceTensor,
//         eager_mode: bool,
//     ) -> DeviceResult<DeviceTensor> {
//         let kernel = if self.function_map.contains_key("transpose") {
//             Function::new(
//                 self.function_map["transpose"].0,
//                 &self.module_map["transpose"].0,
//             )
//         } else {
//             return Err(DeviceError::InvalidPtx);
//         };
//         // #[cfg(feature = "tops_backend")]
//         let input_shape =
//             DeviceBuffer::from_slice(&[arg.shape[0] as i32, arg.shape[1] as i32, 1, 1])?;
//         // #[cfg(feature = "tops_backend")]
//         let matOut = DeviceBuffer::from_slice(&vec![0.0f32; arg.shape[0] * arg.shape[1]])?;

//         #[cfg(feature = "cuda_backend")]
//         let (block_size, grid_a, grid_b) = self.get_block_grid(arg.shape[1], arg.shape[0]);

//         let result: DeviceResult<()> = match (&arg.data, &self.stream) {
//             (Some(data_left), Some(stream)) => match data_left {
//                 DeviceTensorKind::FloatTensor(matA) => unsafe {
//                     #[cfg(feature = "tops_backend")]
//                     let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         input_shape.as_device_ptr()
//                     ));

//                     #[cfg(feature = "cuda_backend")]
//                     let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
//                         matA.as_device_ptr(),
//                         matOut.as_device_ptr(),
//                         arg.shape[0] as u32,
//                         arg.shape[1] as u32,
//                     ));

//                     result
//                 },
//                 _ => {
//                     panic!("Not implemented for other data types!");
//                 }
//             },
//             _ => {
//                 panic!("Invalid data format!");
//             }
//         };

//         if eager_mode {
//             match result {
//                 Ok(_) => match self.synchronize() {
//                     Ok(_) => {
//                         println!("Stream synchronized!");
//                     }
//                     Err(_) => {
//                         panic!("Unable to synchronize kernels!");
//                     }
//                 },
//                 _ => {
//                     panic!("Unable to synchronize kernels!");
//                 }
//             }
//         }

//         match result {
//             Ok(_) => Ok(DeviceTensor {
//                 data: Some(DeviceTensorKind::from(matOut)),
//                 shape: vec![arg.shape[1], arg.shape[0]],
//             }),
//             #[cfg(test)]
//             Err(_e) => {
//                 panic!("Failed to alloc device memory!");
//             }
//             #[cfg(not(test))]
//             Err(e) => {
//                 println!("Failed to alloc device memory!");
//                 Err(e)
//             }
//         }
//     }
// }

// #[cfg(test)]

// mod tests {
//     use std::vec;

//     use super::*;

//     #[test]
//     fn test_matmul_owned() {
//         let exec = DeviceExecutor::new(0);

//         let a = DeviceTensor::ones(vec![17, 23]).unwrap();
//         let b = DeviceTensor::ones(vec![23, 18]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&vec![23.0; 17 * 18], vec![17, 18]).unwrap();

//         let c = exec.matmul_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [17, 18]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_conv2d_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a = DeviceTensor::ones(vec![9, 9]).unwrap();
//         let b = DeviceTensor::fill(vec![3, 3], 0.5f32).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&[4.5f32; 7 * 7], vec![7, 7]).unwrap();

//         let c = exec.conv2d_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [7, 7]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_activation_relu_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

//         exec.activation_inplace(&a, true, "relu".to_string())
//             .unwrap();
//         // assert_eq!(c.ndims(), 2);
//         // assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(a, cref);
//     }

//     #[test]
//     fn test_activation_leaky_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, -0.8, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref =
//             DeviceTensor::from_vec_shape(&[1.0f32, -0.080000006, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
//                 .unwrap();

//         exec.activation_inplace(&a, true, "leaky".to_string())
//             .unwrap();

//         // assert_eq!(a.ndims(), 2);
//         // assert_eq!(a.shape(), [2, 3]);
//         assert_eq!(a, cref);
//     }

//     #[test]
//     fn test_activation_tanh_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(
//             &[
//                 0.7615942f32,
//                 0.9640275,
//                 0.9950547,
//                 0.9993293,
//                 0.99990916,
//                 0.9999877,
//             ],
//             vec![2, 3],
//         )
//         .unwrap();

//         exec.activation_inplace(&a, true, "tanh".to_string())
//             .unwrap();

//         // assert_eq!(c.ndims(), 2);
//         // assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(a, cref);
//     }

//     #[test]
//     fn test_activation_gelu_owned() {
//         let exec = DeviceExecutor::new(0);
//         // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0], vec![2, 4]).unwrap();
//         // let cref = DeviceTensor::from_vec_shape(vec![0.841192f32, 1.9545977, 2.9963627, 3.9999297, 5.0, 6.0, 0.841192f32, 0.841192f32], vec![2, 4]).unwrap();
//         let a = DeviceTensor::from_vec_shape(&[1.0f32; 5 * 5], vec![5, 5]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&[0.841192f32; 5 * 5], vec![5, 5]).unwrap();

//         exec.activation_inplace(&a, true, "gelu".to_string())
//             .unwrap();
//         match &a.data {
//             Some(DeviceTensorKind::FloatTensor(_out)) => {
//                 let _out_host = vec![0.0f32; a.shape[0] * a.shape[1]];
//                 // out.copy_to(&mut out_host);
//                 // for item in out_host {
//                 //     print!("{} ", item)
//                 // }
//             }
//             _ => {
//                 println!("Unable to obtain results!");
//             }
//         }
//         // assert_eq!(c.ndims(), 2);
//         // assert_eq!(c.shape(), [5, 5]);
//         assert_eq!(a, cref);
//     }

//     // #[test]
//     // fn test_matmul_side_effect() {
//     //     let a = DeviceTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
//     //     let b = DeviceTensor::ones(vec![3, 2]);
//     //     let mut c = DeviceTensor::zeros(vec![2, 2]);
//     //     let cref = DeviceTensor::from_vec_shape(vec![6.6000004, 6.6000004, 16.5, 16.5], vec![2, 2]);

//     //     let exec = DeviceExecutor::new();
//     //     exec.matmul_side_effect(&a, &b, &mut c);
//     //     assert_eq!(c.ndims(), 2);
//     //     assert_eq!(c.shape(), [2, 2]);
//     //     assert_eq!(c, cref);
//     // }

//     #[test]
//     fn test_addf32_owned() {
//         let exec = DeviceExecutor::new(0);
//         // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let a = DeviceTensor::from_vec_shape(&vec![1.2f32; 50 * 50], vec![50, 50]).unwrap();
//         let b = DeviceTensor::from_vec_shape(&vec![2.8f32; 50 * 50], vec![50, 50]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&vec![4.0f32; 50 * 50], vec![50, 50]).unwrap();
//         let c = exec.addf32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [50, 50]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_subf32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let b =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&[0.0f32; 6], vec![2, 3]).unwrap();

//         let c = exec.subf32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_mulf32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let b =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&[1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
//             .unwrap();

//         let c = exec.mulf32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_divf32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let b =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape(&[1.0f32; 6], vec![2, 3]).unwrap();
//         let c = exec.divf32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_transpose_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a =
//             DeviceTensor::from_vec_shape(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let cref =
//             DeviceTensor::from_vec_shape(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();
//         let c = exec.transpose_owned(&a, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [3, 2]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_addi32_owned() {
//         let exec = DeviceExecutor::new(0);
//         // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
//         let a = DeviceTensor::from_vec_shape_i32(vec![1i32; 50 * 50], vec![50, 50]).unwrap();
//         let b = DeviceTensor::from_vec_shape_i32(vec![2i32; 50 * 50], vec![50, 50]).unwrap();
//         let cref = DeviceTensor::from_vec_shape_i32(vec![3i32; 50 * 50], vec![50, 50]).unwrap();
//         let c = exec.addi32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [50, 50]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_subi32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a = DeviceTensor::from_vec_shape_i32(vec![3i32, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
//         let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 2, 2, 1, 5], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape_i32(vec![2i32, 0, 1, 2, 4, 1], vec![2, 3]).unwrap();

//         let c = exec.subi32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_muli32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
//         let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 3, 0, 3, 5, 8], vec![2, 3]).unwrap();
//         let cref =
//             DeviceTensor::from_vec_shape_i32(vec![1i32, 6, 0, 12, 25, 48], vec![2, 3]).unwrap();

//         let c = exec.muli32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }

//     #[test]
//     fn test_divi32_owned() {
//         let exec = DeviceExecutor::new(0);
//         let a = DeviceTensor::from_vec_shape_i32(vec![1i32, 4, 3, 4, 5, 6], vec![2, 3]).unwrap();
//         let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 3, 4, 1, 3], vec![2, 3]).unwrap();
//         let cref = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 1, 1, 5, 2], vec![2, 3]).unwrap();
//         let c = exec.divi32_owned(&a, &b, true).unwrap();
//         assert_eq!(c.ndims(), 2);
//         assert_eq!(c.shape(), [2, 3]);
//         assert_eq!(c, cref);
//     }
// }
