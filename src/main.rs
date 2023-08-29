//Example of UHAL for neural network forward pass (on NV GPU & Enflame GCU)
use cust_core::DeviceCopy;
use std::collections::HashMap;

//Import UHAL for common computing interfaces
use ubridge::device_tensor::DeviceTensor;
use uhal::error::DeviceResult;
use uhal::launch;
use uhal::memory::DeviceBufferTrait;
use uhal::module::ModuleTrait;
use uhal::stream::{StreamFlags, StreamTrait};
use uhal::DriverLibraryTrait;
//Tops backend
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;
#[cfg(feature = "tops_backend")]
use tops_backend as tops;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;

fn load_module<'a>(name: &str) -> DeviceResult<Module> {
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/kernels/{}.topsfb", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("{}/kernels/{}.ptx", env!("CARGO_MANIFEST_DIR"), name).to_string();

    Module::from_file(&ptx)
}

struct Layer<'a, T: DeviceCopy> {
    op: &'a str,
    weight: Option<DeviceBuffer<T>>,
    input_size: (usize, usize),
    output_size: (usize, usize),
    out_ref: Option<&'a DeviceBuffer<T>>,
}
pub fn get_block_grid(shape1: usize, shape0: usize) -> (usize, usize, usize) {
    let grid_a: usize = (shape1 + 16 - 1) / 16;
    let grid_b: usize = (shape0 + 16 - 1) / 16;
    return (16, grid_a, grid_b);
}

//A 6-layer neural network forward pass
//Unified interface (UHAL) for CUDA and Tops backend
#[allow(non_snake_case)]
fn network_test() -> DeviceResult<()> {
    let _device = Api::quick_init(0)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const N: usize = 16;
    const K: usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {
            op: "batch_matmul",
            weight: Some(DeviceBuffer::from_slice(&[0.01f32; N * N])?),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is N x N matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "batch_matmul",
            weight: Some(DeviceBuffer::from_slice(&[0.02f32; N * N])?),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is N x N matric for next layer
        Layer::<f32> {
            op: "relu",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "batch_matmul",
            weight: Some(DeviceBuffer::from_slice(&[0.5f32; K * K])?),
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //weight is convolution kernel for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N, N),
            output_size: (N, N),
            out_ref: None,
        }, //out N x N
        Layer::<f32> {
            op: "convolution",
            weight: Some(DeviceBuffer::from_slice(
                &[0.2f32; (N - K + 1) * (N - K + 1)],
            )?),
            input_size: (N, N),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //out (N - K + 1) x (N - K + 1)
        Layer::<f32> {
            op: "batch_matmul",
            weight: Some(DeviceBuffer::from_slice(
                &[0.2f32; (N - K + 1) * (N - K + 1)],
            )?),
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {
            op: "tanh",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //output shape (N - K + 1) * (N - K + 1)
        Layer::<f32> {
            op: "batch_matmul",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, // no weight in the last layer
        Layer::<f32> {
            op: "gelu",
            weight: None,
            input_size: (N - K + 1, N - K + 1),
            output_size: (N - K + 1, N - K + 1),
            out_ref: None,
        }, //output shape (N - K + 1) * (N - K + 1)
    ];

    let mut matA = DeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = DeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = DeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = DeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref: Option<&DeviceBuffer<f32>> = None;
    let mut out_size: Option<(usize, usize)> = None;
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";
            match load_module(function_name) {
                Ok(module) => {
                    let function_namef32 = "activationf32";
                    let kernel = module.get_function(&function_namef32)?;
                    let param = DeviceBuffer::from_slice(&[
                        (layer.input_size.0 * layer.input_size.1) as i32,
                        map_act[layer.op] as i32,
                    ])?;

                    let (block_size, grid_a, grid_b) =
                        get_block_grid(layer.input_size.1, layer.input_size.0);
                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            param.as_device_ptr(),
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            map_act[layer.op]
                        ));

                        result?;
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("Failed to load kernel!");
                }
            }
        } else if layer.op == "batch_matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[
                        1i32,
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                    ])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[
                        1i32,
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                    ])?;

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            layer.output_size.1 as u32
                        ));

                        result?;
                    }
                    std::mem::swap(&mut matA, &mut matOut);
                    match layer.weight {
                        Some(w) => {
                            matB = w;
                        }
                        _ => {
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("\nFailed to load kernel (matmul)!");
                }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let inputShapeA = DeviceBuffer::from_slice(&[
                        layer.input_size.0 as i32,
                        layer.input_size.1 as i32,
                        1i32,
                        1i32,
                    ])?;
                    #[cfg(feature = "tops_backend")]
                    let inputShapeB = DeviceBuffer::from_slice(&[K as i32, K as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            inputShapeA.as_device_ptr(),
                            inputShapeB.as_device_ptr(),
                            channelInfo.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matConvOut.as_device_ptr(),
                            layer.input_size.0 as u32,
                            layer.input_size.1 as u32,
                            K as u32,
                            K as u32
                        ));

                        result?;
                    }

                    std::mem::swap(&mut matA, &mut matConvOut);
                    match layer.weight {
                        Some(w) => {
                            matB = w;
                        }
                        _ => {
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => {
                    panic!("\nFailed to load kernel (convolution)!");
                }
            }
        } else {
            panic!("Operation {} not supported!", layer.op);
        }
    }
    // Wait asynchronous kernels to finish.
    stream.synchronize()?;

    match out_ref {
        Some(out) => {
            let mut out_host = vec![0.0f32; out.len()];
            out.copy_to(&mut out_host[0..out.len()])?;
            match out_size {
                Some(sz) => {
                    let W = sz.0;
                    let H = sz.1;
                    println!("\n\nResults of forward pass******************");
                    for x in 0..H {
                        for y in 0..W {
                            print!("{:.5} ", out_host[x * W + y]);
                        }
                        println!("{}", "");
                    }
                }
                _ => {
                    panic!("Unable to obtain compute result!")
                }
            }
        }
        _ => {
            panic!("Unable to obtain compute result!")
        }
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}

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
fn main() -> DeviceResult<()> {
    match network_test() {
        Ok(()) => {
            println!("\nLaunched network_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch network_test failed.");
            return Err(e);
        }
    }
    println!("\n\nPASSED!\n\n");
    Ok(())
}
