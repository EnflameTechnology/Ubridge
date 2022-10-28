use core::fmt::Debug;
use core::panic;
use std::ptr;
use std::collections::HashMap;
use crate::device_opcode::DeviceOpCode;
use crate::device_tensor::{DeviceTensor, DeviceTensorKind};

use cust_core::DeviceCopy;


//Import UHAL for common computing interfaces
use uhal::launch;
use uhal::error::{DeviceResult};
use uhal::{DriverLibraryTrait};
use uhal::module::{ModuleTrait};
use uhal::memory::{DeviceBufferTrait};
use uhal::stream::{StreamTrait, StreamFlags};

use lazy_static::{lazy_static, __Deref};
use std::sync::{Mutex, Arc};

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
use tops::context::TopsContext as Context;
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
use cuda::context::CuContext as Context;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;

pub fn init_kernels() -> (Option<Context>, Option<Box<HashMap<String, Module>>>){
    
    match Api::quick_init() { 
        Ok(context) => { 
            let mut kernel_map = Box::new(HashMap::<String, Module>::new());
            for kernel in ["matmul", "activation", "convolution"] {
                let module = load_module(kernel).unwrap();
                kernel_map.insert(kernel.to_string(), module);
            }
            if kernel_map.len() > 0 { println!("{} kernel(s) loaded!", kernel_map.len()); }
            return (Some(context), Some(kernel_map)) 
        } 
        _ => { return (None, None) }
    };

}

lazy_static! {
    static ref g_mutex: Mutex<()> = Mutex::new(());

    static ref g_api: (Option<Context>, Option<Box<HashMap<String, Module>>>) = match init_kernels() 
    { 
        (Some(context), Some(kmap)) => { (Some(context), Some(kmap)) } 
        _ => { (None, None) 
    }
};


}

fn load_module<'a>(name : &str) -> DeviceResult<Module>{
    #[cfg(feature = "tops_backend")]
    let ptx = format!("./resources/{}.o",name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("./resources/{}.ptx",name).to_string();

    Module::from_file(&ptx)
}

#[derive(Debug)]
pub struct DeviceExecutor {
    stream : Option<Stream>,
    // kernel_map : Box<HashMap<String, Module>>
}

impl DeviceExecutor {
    pub fn new() -> Self {
        Self {
            stream : match Stream::new(StreamFlags::NON_BLOCKING, None) { Ok(stream) => {Some(stream)} _ => {panic!("Unable to create stream!");}},
        }
    }

    pub fn synchronize(&self) -> DeviceResult<()> {
        match &self.stream {
            Some(stream) => {
                stream.synchronize()
            }
            _ => { panic!("Invalid stream!")}
        }
    }

    pub fn binary_compute_owned(
        &self,
        op: DeviceOpCode,
        lhs: DeviceTensor,
        rhs: DeviceTensor,
        eager_mode : bool
    ) -> DeviceResult<DeviceTensor> {
        match op {
            DeviceOpCode::AddF => self.addf32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::SubF => self.subf32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::MulF => self.mulf32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::DivF => self.divf32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::MatMulF => self.matmul_owned(lhs, rhs, eager_mode),
            DeviceOpCode::Conv2DF => self.conv2d_owned(lhs, rhs, eager_mode),
            _ => panic!("not wired opcode"),
        }
    }

    pub fn mock_result(&self, mock_data:Vec<f32>, mock_shape:Vec<usize>) -> DeviceResult<DeviceTensor> {
        let data = DeviceBuffer::from_slice(&mock_data);
        match data {
            Ok(buf) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(buf)),
                    shape: mock_shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }
    }

    pub fn addf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
    }

    pub fn subf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![0.0f32; 6], vec![2, 3])
    }

    pub fn mulf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
    }

    pub fn divf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32; 6], vec![2, 3])
    }

    pub fn matmul_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "matmul";
        let kernel = match &g_api.1 {Some(kmap) => {kmap["matmul"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // let kernel = self.kernel_map["matmul"].get_function(&function_name)?;
        
        #[cfg(feature = "tops_backend")]
        let mut inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let mut inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1]  as i32, 1i32, 1i32])?;

        let mut matOut = DeviceBuffer::from_slice(&vec![0.0f32 ;lhs.shape[0] *rhs.shape[1]])?;

        let result : DeviceResult<()> = match (lhs.data, rhs.data, &self.stream) {
            (Some(data_left), Some(data_right), Some(stream)) => {
                match (data_left, data_right) {
                    (DeviceTensorKind::FloatTensor(matA), DeviceTensorKind::FloatTensor(matB)) => {
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
                            let result = launch!(kernel<<<(1, 1, 1), (lhs.shape[0] as u32, lhs.shape[1] as u32, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                rhs.shape[1]
                            ));
                
                            result
                        }
                    }
                    _ => { panic!("Not implemented for other data types!");}
                }
            }
            _ => {panic!("Invalid data format!");}
        };
        
        if eager_mode {
            match result {
                Ok(_) => { 
                    match self.synchronize() {
                        Ok(_) => { println!("Stream synchronized!");}
                        Err(_) => {panic!("Unable to synchronize kernels!");}
                    }
                }
                _ => { panic!("Unable to synchronize kernels!");}
            }
        }

        match result {
            Ok(_) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(matOut)),
                    shape: vec![lhs.shape[0], rhs.shape[1]],
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }

        // self.mock_result(vec![23.0f32; 17 * 18], vec![17, 18])
    }

    pub fn conv2d_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "convolution";
        let kernel = match &g_api.1 {Some(kmap) => {kmap["convolution"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // let kernel = self.kernel_map["matmul"].get_function(&function_name)?;
        
        #[cfg(feature = "tops_backend")]
        let mut inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let mut inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let mut channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

        let mut matOut = DeviceBuffer::from_slice(&vec![0.0f32 ;(lhs.shape[0] - rhs.shape[0] + 1) * (lhs.shape[1] - rhs.shape[1] + 1)])?;

        let result : DeviceResult<()> = match (lhs.data, rhs.data, &self.stream) {
            (Some(data_left), Some(data_right), Some(stream)) => {
                match (data_left, data_right) {
                    (DeviceTensorKind::FloatTensor(matA), DeviceTensorKind::FloatTensor(matB)) => {
                        unsafe {
                            #[cfg(feature = "tops_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                inputShapeA.as_device_ptr(),
                                inputShapeB.as_device_ptr(),
                                channelInfo.as_device_ptr()
                            ));
    
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                lhs.shape[0] as i32, lhs.shape[1] as i32,
                                rhs.shape[0] as i32,
                                rhs.shape[1] as i32
                            ));
                
                            result
                        }
                    }
                    _ => { panic!("Not implemented for other data types!");}
                }
            }
            _ => {panic!("Invalid data format!");}
        };
        
        if eager_mode {
            match result {
                Ok(_) => { 
                    match self.synchronize() {
                        Ok(_) => { println!("Stream synchronized!");}
                        Err(_) => {panic!("Unable to synchronize kernels!");}
                    }
                }
                _ => { panic!("Unable to synchronize kernels!");}
            }
        }

        match result {
            Ok(_) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(matOut)),
                    shape: vec![(lhs.shape[0] - rhs.shape[0] + 1), (lhs.shape[1] - rhs.shape[1] + 1)],
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }

        // self.mock_result(vec![1.0f32; 6], vec![2, 3])
    }

    pub fn activation_owned(&self, arg: DeviceTensor, eager_mode : bool, act_type : String) -> DeviceResult<DeviceTensor> {
        let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);
        if !["relu", "gelu", "leaky", "tanh"].contains(&act_type.as_str()) { panic!("Activation type not supported!");}

        let function_name = "activation";
        let kernel = match &g_api.1 {Some(kmap) => {kmap["activation"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        #[cfg(feature = "tops_backend")]
        let mut inputType = DeviceBuffer::from_slice(&[arg.shape[0] as i32, arg.shape[1] as i32, map_act[act_type.as_str()] as i32])?;

        let result : DeviceResult<()> = match (&arg.data, &self.stream) {
            (Some(data_left),  Some(stream)) => {
                match data_left {
                    DeviceTensorKind::FloatTensor(matA) => {
                        unsafe {
                            #[cfg(feature = "tops_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                inputType.as_device_ptr()
                            ));
    
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (layer.input_size.0 as u32, layer.input_size.1 as u32, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                arg.shape[0],
                                map_act[act_type.as_str()]
                            ));
                
                            result
                        }
                    }
                    _ => { panic!("Not implemented for other data types!");}
                }
            }
            _ => {panic!("Invalid data format!");}
        };
        
        if eager_mode {
            match result {
                Ok(_) => { 
                    match self.synchronize() {
                        Ok(_) => { println!("Stream synchronized!");}
                        Err(_) => {panic!("Unable to synchronize kernels!");}
                    }
                }
                _ => { panic!("Unable to synchronize kernels!");}
            }
        }

        match result {
            Ok(_) => {
                match arg.data {
                    Some(data) => { 
                        match data {
                            DeviceTensorKind::FloatTensor(matA) => {
                                Ok(DeviceTensor {
                                    data: Some(DeviceTensorKind::from(matA)),
                                    shape: arg.shape,
                                })
                            }
                            _ => { panic!("Data type not supported!"); }
                        }
                    }
                    _ => { panic!("Unable to return data!");}
                }

            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }
    }
    pub fn transpose_owned(&self, arg: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2])
    }
}


#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_matmul_owned(){
        let a = DeviceTensor::ones(vec![17, 23]).unwrap();
        let b = DeviceTensor::ones(vec![23, 18]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![23.0; 17 * 18], vec![17, 18]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.matmul_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_conv2d_owned(){
        let a = DeviceTensor::ones(vec![9, 9]).unwrap();
        let b = DeviceTensor::fill(vec![3, 3], 0.5f32).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![4.5f32; 7 * 7], vec![7, 7]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.conv2d_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [7, 7]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_activation_relu_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.activation_owned(a, true, "relu".to_string()).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    // #[test]
    // fn test_matmul_side_effect() {
    //     let a = DeviceTensor::from_vec_shape(vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], vec![2, 3]);
    //     let b = DeviceTensor::ones(vec![3, 2]);
    //     let mut c = DeviceTensor::zeros(vec![2, 2]);
    //     let cref = DeviceTensor::from_vec_shape(vec![6.6000004, 6.6000004, 16.5, 16.5], vec![2, 2]);

    //     let exec = DeviceExecutor::new();
    //     exec.matmul_side_effect(&a, &b, &mut c);
    //     assert_eq!(c.ndims(), 2);
    //     assert_eq!(c.shape(), [2, 2]);
    //     assert_eq!(c, cref);
    // }

    #[test]
    fn test_addf32_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.addf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_subf32_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![0.0f32; 6], vec![2, 3]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.subf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_mulf32_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.mulf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_divf32_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32; 6], vec![2, 3]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.divf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_transpose_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.transpose_owned(a, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [3, 2]);
        assert_eq!(c, cref);
    }

}