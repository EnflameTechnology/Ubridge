<div align="center">
<h2 align="center">UHAL Bridge (ubridge) - Bridge between computing frameworks and UHAL. </h2>
<br />
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg" /><br>
<br>
This Rust crate serve as the bridge between "computing frameworks", such as Chopper and Candle and "UHAL". <br> The computing instructions will be dispatched to the corresponding devices through: ubridge -> UHAL -> CUDA driver or Tops driver.
</div>

## UPDATE KERNELS

The gcu kernels written in TopsCC will be build automatically when compiling Ubridge or Candle-GCU project.

You may also manually build them if there are any changes:
```
cd gcu_kernels && cargo build
```

## Items
device_tensor.rs: higher level abstraction of device tensor. 

device_executor.rs: execution engine and kernel management. 

device_opcode.rs: definition of operators that currently supported.

gcu_device.rs: abstraction of GCU device for Candle.

gcu_slice.rs: used for tensor slicing for candle-gcu.

gcu_launch.rs: gcu kernel launch for candle-gcu.

tests/*: samples of UHAL, ubridge.

main.rs: entry for samples (executed by **cargo run**).

kernels: CUDA/GCU kernels.

## The entire workflow （for Candle-GCU):
Candle Models **->** 
Candle-nn / Candle-core  **->** 
GCU Backend **->** 
UBridge (http://git.enflame.cn/guoqing.bao/ubridge) **->** 
UHAL (http://git.enflame.cn/guoqing.bao/UHHI/) **->** 
Concreate backend (CUDA/Tops) **->**
Drivers (CUDA/Tops) **->**
Nvidia GPU/Enflame GCU

## The entire workflow （for Chopper）:
Frontend (Pytorch, Tensorflow, Jax) scripts **->** 
Chopper (_Compiler -> Chopper Runtime -> Raptors_) **->** 
UBridge (http://git.enflame.cn/guoqing.bao/ubridge) **->** 
UHAL (http://git.enflame.cn/guoqing.bao/UHHI/) **->** 
Concreate backend (CUDA/Tops) **->**
Drivers (CUDA/Tops) **->**
Nvidia GPU/Enflame GCU

## Example of UHAL 
#### A 6-layer neural network forward pass on GPU/GCU

``` rust
//Import UHAL for common computing interfaces
use uhal::launch;
use uhal::error::{DeviceResult};
use uhal::{DriverLibraryTrait};
use uhal::module::{ModuleTrait};
use uhal::memory::{DeviceBufferTrait};
use uhal::stream::{StreamTrait, StreamFlags};

//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

fn load_module<'a>(name : &str) -> DeviceResult<Module>{
    #[cfg(feature = "tops_backend")]
    let ptx = format!("./kernels/{}.o",name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("./kernels/{}.ptx",name).to_string();

    Module::from_file(&ptx)
}

struct Layer<'a, T: DeviceCopy> {
    op : &'a str,
    weight : Option<DeviceBuffer<T>>,
    input_size : (usize, usize),
    output_size : (usize, usize),
    out_ref : Option<&'a DeviceBuffer<T>>
}

fn network_test() -> DeviceResult<()> {
    let _ctx = Api::quick_init()?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    const N : usize = 16;
    const K : usize = 3;

    //Neural network layers: matmul(tanh act) -> matmul(relu act) -> matmul(tanh act) -> convolution(3x3 kernel, tanh act) -> matmul(tanh act) -> matmul(leaky act)
    let layers = vec![
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.01f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.02f32; N * N])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is N x N matric for next layer
        Layer::<f32> {op : "relu", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.5f32; K * K])?), input_size : (N, N), output_size : (N, N), out_ref : None}, //weight is convolution kernel for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N, N), output_size : (N, N), out_ref : None}, //out N x N

        Layer::<f32> {op : "convolution", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N, N), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None},  //out (N - K + 1) x (N - K + 1)
        
        Layer::<f32> {op : "matmul", weight: Some(DeviceBuffer::from_slice(&[0.2f32; (N - K + 1) * (N - K + 1)])?), input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //weight is (N - K + 1) * (N - K + 1) matric for next layer
        Layer::<f32> {op : "tanh", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)

        Layer::<f32> {op : "matmul", weight: None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, // no weight in the last layer
        Layer::<f32> {op : "gelu", weight : None, input_size : (N - K + 1, N - K + 1), output_size : (N - K + 1, N - K + 1), out_ref : None}, //output shape (N - K + 1) * (N - K + 1)
    ];

    let mut matA = DeviceBuffer::from_slice(&[0.5f32; N * N])?;
    let mut matB = DeviceBuffer::from_slice(&[0.1f32; N * N])?;
    let mut matOut = DeviceBuffer::from_slice(&[0.0f32; N * N])?;
    let mut matConvOut = DeviceBuffer::from_slice(&[0.0f32; (N - K + 1) * (N - K + 1)])?;

    let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);

    let mut out_ref : Option<&DeviceBuffer<f32>> = None;
    let mut out_size : Option<(usize, usize)> = None;
    for layer in layers {
        if ["relu", "gelu", "leaky", "tanh"].contains(&layer.op) {
            let function_name = "activation";

            #[cfg(feature = "tops_backend")]
            let mut inputType = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, map_act[layer.op] as i32])?;

            match load_module(function_name) {
                Ok(module) => {
                    let kernel = module.get_function(&function_name)?;
                    unsafe {
                        #[cfg(feature = "tops_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            inputType.as_device_ptr()
                        ));

                        #[cfg(feature = "cuda_backend")]
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            layer.output_size.0,
                            map_act[layer.op]
                        ));

                        result?;
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("Failed to load kernel!"); }
            }
        } else if layer.op == "matmul" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let mut inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let mut inputShapeB = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;

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
                        let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                            matA.as_device_ptr(),
                            matB.as_device_ptr(),
                            matOut.as_device_ptr(),
                            layer.output_size.0
                        ));

                        result?;
                    }
                    std::mem::swap(&mut matA, &mut matOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);
                }
                _ => { panic!("\nFailed to load kernel (matmul)!"); }
            }
        } else if layer.op == "convolution" {
            match load_module(layer.op) {
                Ok(module) => {
                    let kernel = module.get_function(&layer.op)?;

                    #[cfg(feature = "tops_backend")]
                    let mut inputShapeA = DeviceBuffer::from_slice(&[layer.input_size.0 as i32, layer.input_size.1 as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let mut inputShapeB = DeviceBuffer::from_slice(&[K as i32, K as i32, 1i32, 1i32])?;
                    #[cfg(feature = "tops_backend")]
                    let mut channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

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
                            layer.input_size.0 as i32, layer.input_size.1 as i32,
                            K as i32,
                            K as i32
                        ));

                        result?;
                    }

                    std::mem::swap(&mut matA, &mut matConvOut);
                    match layer.weight {
                        Some(w) => { matB = w;}
                        _ => { 
                            // if idx < len - 1 { println!("Failed to get weight!"); break; }
                        }
                    }
                    out_ref = Some(&matA);
                    out_size = Some(layer.output_size);

                }
                _ => { panic!("\nFailed to load kernel (convolution)!"); }
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
                _ => { panic!("Unable to obtain compute result!") }
            }

        }
        _ => { panic!("Unable to obtain compute result!")}
    }

    println!("\nLaunched compute kernel successfully.");

    Ok(())
}

fn main() -> DeviceResult<()> {
    println!("******************\ninfo: start uhal network test!\n");
    
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
```

### The output of the forward pass should be:
 _(Same on Nvidia GPU and Enflame GCU)_
```
Results of forward pass******************
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 0.97597 
```

#### External dependencies
**Computing on GCU**: 
Enflame GCU Driver, GCU Runtime 2.0/3.0, Enflame T20 card_

**Computing on GPU**: 
_CUDA 11.3, Nvidia Driver, Nvidia GPU card_
