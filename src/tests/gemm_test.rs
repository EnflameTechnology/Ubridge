//Example of UHAL for neural network forward pass (on NV GPU & Enflame GCU)
use cust_core::DeviceCopy;
use std::collections::HashMap;
use std::os::raw::c_void;

//Import UHAL for common computing interfaces

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

use crate::gemm_tuner::{AtenGemmTuner, AtenGemmInfo, AtenGemmTune, GEMM_OP_PARAS};


fn load_module<'a>(name: &str) -> DeviceResult<Module> {
    #[cfg(not(feature = "scorpio"))]
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/kernels/pavo/{}.topsfb", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "scorpio")]
    let ptx = format!("{}/kernels/scorpio/{}.topsfb", env!("CARGO_MANIFEST_DIR"), name).to_string();

    Module::from_file(&ptx)
}

struct Layer<'a, T: DeviceCopy> {
    op: &'a str,
    weight: Option<DeviceBuffer<T>>,
    input_size: (i64, i64, i64),
    output_size: (i64, i64, i64),
    out_ref: Option<&'a DeviceBuffer<T>>,
}

#[allow(non_snake_case)]
pub fn gemm_test() -> DeviceResult<()> {
    let _device = Api::quick_init(0)?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    const B: usize = 1;
    const M: usize = 1;
    const K: usize = 4096;
    const N: usize = 4096;

    let mut info = AtenGemmInfo::default();
    info.batch = B as i64;
    info.M = M as i64;
    info.K = K as i64;
    info.N = N as i64;
    info.batch = 16;
    info.is_batch = true;
    let mut tune = AtenGemmTune::default();
    let tuner = AtenGemmTuner::new();
    tuner.tuner(&info, &mut tune);

    let param = GEMM_OP_PARAS::new(&info, &tune);

    let lhs = vec![0.5f32; B * M * K];
    let rhs = vec![0.5f32; B * K * N];
    let out = vec![0.0f32; B * M * K];
    let bias = vec![0.0f32; N];

    let matA = DeviceBuffer::from_slice(&lhs)?;
    let matOut = DeviceBuffer::from_slice(&out)?;
    let matBias = DeviceBuffer::from_slice(&bias)?;

    let layers = vec![
        Layer::<f32> {
            op: "gemm",
            weight: Some(DeviceBuffer::from_slice(&rhs)?),
            input_size: (B as i64, M as i64, K as i64),
            output_size: (B as i64, M as i64, N as i64),
            out_ref: None,
        }, //weight is N x N matric for next layer
    ];
  

    let mut out_ref: Option<&DeviceBuffer<f32>> = None;
    let mut out_size: Option<(i64, i64, i64)> = None;
    for layer in layers {
        if layer.op == "gemm" {
            match load_module(layer.op) {
                Ok(module) => {
                    let function_namef32 = "gemm_f32";

                    let kernel = module.get_function(&function_namef32)?;
                    match layer.weight {
                        Some(w) => {
                            unsafe {
                                // let vaddress = std::mem::transmute::<*mut *mut c_void, *mut c_void>(ptr_param as *mut *mut c_void);

                                #[cfg(feature = "tops_backend")]
                                let result = launch!(kernel<<<(1, 1, 1), (12, 1, 1), 0, stream>>>(
                                    matA.as_device_ptr(),
                                    w.as_device_ptr(),
                                    matOut.as_device_ptr(),
                                    matBias.as_device_ptr(),
                                    param.input_dtype, B, M, K, N,
                                    param.lhs_multicore, param.rhs_multicore, param.batch_multicore,
                                    param.lhs_transpose, param.rhs_transpose,
                                    param.alpha, param.beta, param.addmm_beta, param.bias,
                                    param.sip_m, param.sip_k, param.sip_n
                                ));
        
                                result?;
                            }
                            out_ref = Some(&matOut);
                            out_size = Some(layer.output_size);
                        }
                        _ => {
                        }

                    }

                }
                _ => {
                    panic!("\nFailed to load kernel (matmul)!");
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
                    let B1 = sz.0;
                    let M1 = sz.1;
                    let N1 = sz.2;
                    println!("\n\nResults of forward pass******************");
                    for b in 0..B1 {
                        for x in 0..N1 {
                            for y in 0..M1 {
                                print!("{:.5} ", out_host[(b*M1*N1 + x * M1 + y) as usize]);
                            }
                            println!("{}", "");
                        }
                        println!("\n\n******************");

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