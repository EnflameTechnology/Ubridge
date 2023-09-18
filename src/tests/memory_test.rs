#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;

use crate::device_tensor::DeviceTensor;
use uhal::{error::DeviceResult, DriverLibraryTrait};

fn test_device_tensor(rawptr: *const f32, dst: &mut Vec<f32>, b: usize, m: usize, k: usize) {
    let ltensor = DeviceTensor::from_pointer(rawptr, b * m * k, vec![b, m, k]).unwrap();
    ltensor.to_cpu(dst);

    // assert!(dst==lhs);
}

fn memory_test() -> DeviceResult<()> {
    let _device = Api::quick_init(0)?;

    println!("******************\ninfo: start uhal network test!\n");
    let b = 1;
    let m = 4095;
    let k = 11007;
    let lhs = vec![1.0f32; b * m * k];
    let rawptr = lhs.as_ptr().cast::<f32>();
    let mut dst: Vec<f32> = Vec::with_capacity(b * m * k);

    let lhs1 = vec![2.0f32; b * m * (k / 2)];
    let rawptr1 = lhs1.as_ptr().cast::<f32>();
    let mut dst1: Vec<f32> = Vec::with_capacity(b * m * (k / 2));

    let lhs2 = vec![2.11f32; b * 4 * m * (k / 2)];
    let rawptr2 = lhs2.as_ptr().cast::<f32>();
    let mut dst2: Vec<f32> = Vec::with_capacity(b * 4 * m * (k / 2));

    let lhs3 = vec![2.33f32; b * m * (k / 10)];
    let rawptr3 = lhs3.as_ptr().cast::<f32>();
    let mut dst3: Vec<f32> = Vec::with_capacity(b * m * (k / 10));

    unsafe {
        dst.set_len(b * m * k);
    }
    unsafe {
        dst1.set_len(b * m * (k / 2));
    }
    unsafe {
        dst2.set_len(b * 4 * m * (k / 2));
    }
    unsafe {
        dst3.set_len(b * m * (k / 10));
    }

    for i in 0..100 {
        test_device_tensor(rawptr, &mut dst, b, m, k);
        test_device_tensor(rawptr1, &mut dst1, b, m, k / 2);
        test_device_tensor(rawptr2, &mut dst2, b * 4, m, k / 2);
        test_device_tensor(rawptr3, &mut dst3, b, m, k / 10);

        println!("Processed {}", i);
    }

    Ok(())
}