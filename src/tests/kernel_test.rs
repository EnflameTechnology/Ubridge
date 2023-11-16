#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

#[cfg(feature = "tops_backend")]
use tops_backend as tops;


use crate::device_tensor::DeviceTensor;
use crate::device_executor::DeviceExecutor;
use uhal::{error::DeviceResult};

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
    unsafe { dst.set_len(b * m * n); }

    // if executor.has_function("transposed_matmul".to_string(), "transposed_matmul".to_string()) {
    //     let out = executor.transposed_matmul_owned(&ltensor, &rtensor, true).unwrap();
    //     out.to_cpu(&mut dst);
    // }

    Ok(())
}