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
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

use crate::device_tensor::DeviceTensor;
use uhal::error::DeviceResult;

pub fn kernel_test() -> DeviceResult<()> {
    println!("******************\ninfo: start kernel test!\n");
    let b = 1;
    let m = 128;
    let k = 4096;
    let n = 4096;
    let lhs = vec![1.0f32; b * m * k];
    let rhs = vec![0.5f32; b * k * n];
    let _ltensor = DeviceTensor::from_vec_shape(&lhs, vec![b, m, k]).unwrap();
    let _rtensor = DeviceTensor::from_vec_shape(&rhs, vec![b, k, n]).unwrap();

    let mut dst: Vec<f32> = Vec::with_capacity(b * m * n);
    unsafe {
        dst.set_len(b * m * n);
    }

    // if executor.has_function("transposed_matmul".to_string(), "transposed_matmul".to_string()) {
    //     let out = executor.transposed_matmul_owned(&ltensor, &rtensor, true).unwrap();
    //     out.to_cpu(&mut dst);
    // }

    Ok(())
}
