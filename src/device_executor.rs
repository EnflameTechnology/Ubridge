use core::fmt::Debug;
use core::panic;
use std::ptr;
use std::collections::HashMap;
use crate::device_opcode::DeviceOpCode;
use crate::device_tensor::{DeviceTensor, DeviceTensorKind};
use std::sync::Once;
use cust_core::DeviceCopy;


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
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops::context::TopsContext as Context;
#[cfg(feature = "tops_backend")]
use tops::module::TopsModule as Module;
#[cfg(feature = "tops_backend")]
use tops::TopsApi as Api;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::stream::CuStream as Stream;
#[cfg(feature = "cuda_backend")]
use cuda::context::CuContext as Context;
#[cfg(feature = "cuda_backend")]
use cuda::module::CuModule as Module;
#[cfg(feature = "cuda_backend")]
use cuda::CuApi as Api;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;

static mut G_KERNEL: (Option<Box<HashMap<String, Module>>>, Option<Context>) =  (None, None);
static INIT: Once = Once::new();

pub fn init_api() -> Option<Context>{
    match Api::quick_init() { 
        Ok(context) => { 
            return Some(context)
        } 
        _ => { return None }
    };
}

pub fn init_kernels() -> (Option<Box<HashMap<String, Module>>>, Option<Context>){
    
    match init_api()  { 
        Some(context) => { 
            let mut kernel_map = Box::new(HashMap::<String, Module>::new());
            for kernel in ["matmul", "activation", "convolution", "transpose", "element", "elementi32"] {
                let module = load_module(kernel).unwrap();
                kernel_map.insert(kernel.to_string(), module);
            }
            if kernel_map.len() > 0 { println!("{} kernel(s) loaded!", kernel_map.len()); }
            return (Some(kernel_map), Some(context))
        } 
        _ => { return (None, None) }
    };

}

fn get_kernels() -> &'static (Option<Box<HashMap<String, Module>>>, Option<Context>) {
    unsafe {
        INIT.call_once(|| {
            G_KERNEL = init_kernels();
        });
        &G_KERNEL
    }
}


fn load_module<'a>(name : &str) -> DeviceResult<Module>{
    #[cfg(feature = "tops_backend")]
    let ptx = format!("{}/resources/{}.o", env!("CARGO_MANIFEST_DIR"), name).to_string();

    #[cfg(feature = "cuda_backend")]
    let ptx = format!("{}/resources/{}.ptx", env!("CARGO_MANIFEST_DIR"), name).to_string();

    println!("{}", ptx);

    Module::from_file(&ptx)
}

#[derive(Debug)]
pub struct DeviceExecutor {
    stream : Option<Stream>,
    // kernel_map : Box<HashMap<String, Module>>
    kernel_map: Option<&'static Box<HashMap<String, Module>>>, 
    context : Option<&'static Context>

}

impl DeviceExecutor {
    pub fn new() -> Self {
        match get_kernels() {
            (Some(_kernel_map), Some(_context)) => {
                Self {
                    context : Some(_context),
                    kernel_map: Some(_kernel_map),
                    stream : match Stream::new(StreamFlags::NON_BLOCKING, None) { Ok(stream) => {Some(stream)} _ => {panic!("Unable to create stream!");}},
                }
            }
            _ => panic!("Load kernels failed!")
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
    
    pub fn unary_compute_owned(        
        &self,
        op: DeviceOpCode,
        arg: DeviceTensor,
        eager_mode : bool) -> DeviceResult<DeviceTensor> {
        match op {
            DeviceOpCode::RELU => self.activation_owned(arg, eager_mode, "relu".to_string()),
            DeviceOpCode::GELU => self.activation_owned(arg, eager_mode, "gelu".to_string()),
            DeviceOpCode::LEAKY => self.activation_owned(arg, eager_mode, "leaky".to_string()),
            DeviceOpCode::TANH => self.activation_owned(arg, eager_mode, "tanh".to_string()),
            DeviceOpCode::Transpose => self.transpose_owned(arg, eager_mode),
            _ => panic!("Not supported operation!"),
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

            DeviceOpCode::AddI => self.addi32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::SubI => self.subi32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::MulI => self.muli32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::DivI => self.divi32_owned(lhs, rhs, eager_mode),
            DeviceOpCode::MatMulF => self.matmul_owned(lhs, rhs, eager_mode),
            DeviceOpCode::Conv2DF => self.conv2d_owned(lhs, rhs, eager_mode),
            _ => panic!("Not supported operation!"),
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
        // self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
        self.elementf32_owned(lhs, rhs, 0i32, eager_mode)

    }

    pub fn subf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![0.0f32; 6], vec![2, 3])
        self.elementf32_owned(lhs, rhs, 1i32, eager_mode)

    }

    pub fn mulf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
        self.elementf32_owned(lhs, rhs, 2i32, eager_mode)
    }

    pub fn divf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![1.0f32; 6], vec![2, 3])
        self.elementf32_owned(lhs, rhs, 3i32, eager_mode)
    }

    
    pub fn addi32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
        self.elementi32_owned(lhs, rhs, 0i32, eager_mode)

    }

    pub fn subi32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![0.0f32; 6], vec![2, 3])
        self.elementi32_owned(lhs, rhs, 1i32, eager_mode)

    }

    pub fn muli32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
        self.elementi32_owned(lhs, rhs, 2i32, eager_mode)
    }

    pub fn divi32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        // self.mock_result(vec![1.0f32; 6], vec![2, 3])
        self.elementi32_owned(lhs, rhs, 3i32, eager_mode)
    }

    pub fn get_block_grid(&self, shape1:usize, shape0:usize) -> (usize, usize, usize) {
        let grid_a : usize = (shape1 + 16 - 1) / 16;
        let grid_b : usize = (shape0 + 16 - 1) / 16;
        return (16, grid_a, grid_b)
    }

    #[allow(non_snake_case)]
    pub fn elementf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, tp : i32, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "element";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["element"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // let kernel = self.kernel_map["matmul"].get_function(&function_name)?;
        let size : usize = lhs.shape.iter().product();
        let matOut = DeviceBuffer::from_slice(&vec![0.0f32; size])?;
        let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

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
                                size as i32,
                                tp as i32
                            ));
                
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                lhs.shape[0] as u32,
                                lhs.shape[1] as u32,
                                tp as u32
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
                    shape: lhs.shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }
    }

    #[allow(non_snake_case)]
    pub fn elementi32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, tp : i32, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "elementi32";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["elementi32"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        let size : usize = lhs.shape.iter().product();
        let matOut = DeviceBuffer::from_slice(&vec![0i32; size])?;
        let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

        let result : DeviceResult<()> = match (lhs.data, rhs.data, &self.stream) {
            (Some(data_left), Some(data_right), Some(stream)) => {
                match (data_left, data_right) {
                    (DeviceTensorKind::Int32Tensor(matA), DeviceTensorKind::Int32Tensor(matB)) => {
                        unsafe {
                            #[cfg(feature = "tops_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                size as i32,
                                tp as i32
                            ));
                
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                lhs.shape[0] as u32,
                                lhs.shape[1] as u32,
                                tp as u32
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
                    shape: lhs.shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }
    }

    //Maximum input size 512 x 512 supported!
    #[allow(non_snake_case)]
    pub fn matmul_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "matmul";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["matmul"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // let kernel = self.kernel_map["matmul"].get_function(&function_name)?;
        
        #[cfg(feature = "tops_backend")]
        let inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1]  as i32, 1i32, 1i32])?;

        let matOut = DeviceBuffer::from_slice(&vec![0.0f32 ;lhs.shape[0] *rhs.shape[1]])?;
        let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

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
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                lhs.shape[0] as u32,
                                lhs.shape[1] as u32,
                                rhs.shape[1] as u32
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

    #[allow(non_snake_case)]
    pub fn conv2d_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "convolution";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["convolution"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // let kernel = self.kernel_map["matmul"].get_function(&function_name)?;
        
        #[cfg(feature = "tops_backend")]
        let inputShapeA = DeviceBuffer::from_slice(&[lhs.shape[0] as i32, lhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let inputShapeB = DeviceBuffer::from_slice(&[rhs.shape[0] as i32, rhs.shape[1] as i32, 1i32, 1i32])?;
        #[cfg(feature = "tops_backend")]
        let channelInfo = DeviceBuffer::from_slice(&[1i32, 1i32, 1i32, 1i32])?;

        let matOut = DeviceBuffer::from_slice(&vec![0.0f32 ;(lhs.shape[0] - rhs.shape[0] + 1) * (lhs.shape[1] - rhs.shape[1] + 1)])?;
        let (block_size, grid_a, grid_b) = self.get_block_grid(rhs.shape[1], lhs.shape[0]);

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
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                matB.as_device_ptr(),
                                matOut.as_device_ptr(),
                                lhs.shape[0] as i32, 
                                lhs.shape[1] as i32,
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

    #[allow(non_snake_case)]
    pub fn activation_owned(&self, arg: DeviceTensor, eager_mode : bool, act_type : String) -> DeviceResult<DeviceTensor> {
        let map_act = HashMap::from([("relu", 0), ("gelu", 1), ("leaky", 2), ("tanh", 3)]);
        if !["relu", "gelu", "leaky", "tanh"].contains(&act_type.as_str()) { panic!("Activation type not supported!");}

        let function_name = "activation";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["activation"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        let size : usize = arg.shape.iter().product();
        let (block_size, grid_a, grid_b) = self.get_block_grid(arg.shape[1], arg.shape[0]);

        let result : DeviceResult<()> = match (&arg.data, &self.stream) {
            (Some(data_left),  Some(stream)) => {
                match data_left {
                    DeviceTensorKind::FloatTensor(matA) => {
                        unsafe {
                            #[cfg(feature = "tops_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                size as i32,
                                map_act[act_type.as_str()] as i32
                            ));
    
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                arg.shape[0] as u32,
                                arg.shape[1] as u32,
                                map_act[act_type.as_str()] as i32
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

    //Maximum input size 512 x 512 supported!
    #[allow(non_snake_case)]
    pub fn transpose_owned(&self, arg: DeviceTensor, eager_mode : bool) -> DeviceResult<DeviceTensor> {
        let function_name = "transpose";
        let kernel = match &self.kernel_map {Some(kmap) => {kmap["transpose"].get_function(&function_name)?} _=> {panic!("Unable to use kernel!");}};
        // #[cfg(feature = "tops_backend")]
        let input_shape = DeviceBuffer::from_slice(&[arg.shape[0] as i32, arg.shape[1] as i32, 1, 1])?;
        // #[cfg(feature = "tops_backend")]
        let matOut = DeviceBuffer::from_slice(&vec![0.0f32; arg.shape[0] * arg.shape[1]])?;
        let (block_size, grid_a, grid_b) = self.get_block_grid(arg.shape[1], arg.shape[0]);

        let result : DeviceResult<()> = match (&arg.data, &self.stream) {
            (Some(data_left),  Some(stream)) => {
                match data_left {
                    DeviceTensorKind::FloatTensor(matA) => {
                        unsafe {
                            #[cfg(feature = "tops_backend")]
                            let result = launch!(kernel<<<(1, 1, 1), (1, 1, 1), 0, stream>>>(
                                matA.as_device_ptr(),
                                matOut.as_device_ptr(),
                                input_shape.as_device_ptr()
                            ));
    
                            #[cfg(feature = "cuda_backend")]
                            let result = launch!(kernel<<<(grid_a as u32, grid_b as u32), (block_size as u32, block_size as u32), 0, stream>>>(
                                matA.as_device_ptr(),
                                matOut.as_device_ptr(),
                                arg.shape[0] as u32,
                                arg.shape[1] as u32,  
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
                    shape: vec![arg.shape[1], arg.shape[0]],
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(e) => { println!("Failed to alloc device memory!"); Err(e) }
        }
    }
}


#[cfg(test)]

mod tests {
    use std::vec;

    use super::*;

    #[test]
    fn test_matmul_owned(){
        let exec = DeviceExecutor::new();

        let a = DeviceTensor::ones(vec![17, 23]).unwrap();
        let b = DeviceTensor::ones(vec![23, 18]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![23.0; 17 * 18], vec![17, 18]).unwrap();
        
        let c = exec.matmul_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_conv2d_owned(){
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::ones(vec![9, 9]).unwrap();
        let b = DeviceTensor::fill(vec![3, 3], 0.5f32).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![4.5f32; 7 * 7], vec![7, 7]).unwrap();

        let c = exec.conv2d_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [7, 7]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_activation_relu_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let c = exec.activation_owned(a, true, "relu".to_string()).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_activation_leaky_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, -0.8, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, -0.080000006, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let c = exec.activation_owned(a, true, "leaky".to_string()).unwrap();

        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_activation_tanh_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![0.7615942f32, 0.9640275, 0.9950547, 0.9993293, 0.99990916, 0.9999877], vec![2, 3]).unwrap();

        let c = exec.activation_owned(a, true, "tanh".to_string()).unwrap();

        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_activation_gelu_owned() {
        let exec = DeviceExecutor::new();
        // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 1.0], vec![2, 4]).unwrap();
        // let cref = DeviceTensor::from_vec_shape(vec![0.841192f32, 1.9545977, 2.9963627, 3.9999297, 5.0, 6.0, 0.841192f32, 0.841192f32], vec![2, 4]).unwrap();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32; 5*5], vec![5, 5]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![0.841192f32; 5*5], vec![5, 5]).unwrap();

        let c = exec.activation_owned(a, true, "gelu".to_string()).unwrap();
        match &c.data {
            Some(data) => {
                match data {
                    DeviceTensorKind::FloatTensor(out) => {
                        let mut out_host = vec![0.0f32; c.shape[0] * c.shape[1]];
                        out.copy_to(&mut out_host);
                        for item in out_host {print!("{} ", item)};
                    }
                    _ => { println!("Unable to obtain results!");}
                }
            }
            _ => {println!("Unable to obtain results!");}
        }
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [5, 5]);
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
        let exec = DeviceExecutor::new();
        // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let a = DeviceTensor::from_vec_shape(vec![1.2f32; 50*50], vec![50, 50]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![2.8f32; 50*50], vec![50, 50]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![4.0f32; 50*50], vec![50, 50]).unwrap();
        let c = exec.addf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [50, 50]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_subf32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![0.0f32; 6], vec![2, 3]).unwrap();

        let c = exec.subf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_mulf32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3]).unwrap();

        let c = exec.mulf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_divf32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32; 6], vec![2, 3]).unwrap();
        let c = exec.divf32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_transpose_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();
        let c = exec.transpose_owned(a, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [3, 2]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_addi32_owned() {
        let exec = DeviceExecutor::new();
        // let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let a = DeviceTensor::from_vec_shape_i32(vec![1i32; 50*50], vec![50, 50]).unwrap();
        let b = DeviceTensor::from_vec_shape_i32(vec![2i32; 50*50], vec![50, 50]).unwrap();
        let cref = DeviceTensor::from_vec_shape_i32(vec![3i32; 50*50], vec![50, 50]).unwrap();
        let c = exec.addi32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [50, 50]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_subi32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape_i32(vec![3i32, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 2, 2, 1, 5], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape_i32(vec![2i32, 0, 1, 2, 4, 1], vec![2, 3]).unwrap();

        let c = exec.subi32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_muli32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 3, 0, 3, 5, 8], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape_i32(vec![1i32, 6, 0, 12, 25, 48], vec![2, 3]).unwrap();

        let c = exec.muli32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_divi32_owned() {
        let exec = DeviceExecutor::new();
        let a = DeviceTensor::from_vec_shape_i32(vec![1i32, 4, 3, 4, 5, 6], vec![2, 3]).unwrap();
        let b = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 3, 4, 1, 3], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape_i32(vec![1i32, 2, 1, 1, 5, 2], vec![2, 3]).unwrap();
        let c = exec.divi32_owned(a, b, true).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }
}