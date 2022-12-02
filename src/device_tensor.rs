use std::fmt::Debug;
use float_eq::float_eq;

//Import UHAL for common computing interfaces
use uhal::error::{DeviceResult};
use uhal::memory::{DeviceBufferTrait};
use uhal::context::CurrentContextTrait;
use crate::device_executor::G_KERNEL;
//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use cuda::context::TopsCurrentContext as CurrentContext;

//Cuda backend
#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer as DeviceBuffer;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CopyDestination;
#[cfg(feature = "cuda_backend")]
use cuda::context::CuCurrentContext as CurrentContext;

// TODO consider hide TensorKind, and expose a into_raw_vec for BlasTensor
#[derive(Debug, Clone)]
pub enum DeviceTensorKind {
    FloatTensor(DeviceBuffer<f32>),
    DoubleTensor(DeviceBuffer<f64>),
    Int32Tensor(DeviceBuffer<i32>),
    Int8Tensor(DeviceBuffer<i8>),
}

#[derive(Debug, Clone)]
pub struct DeviceTensor {
    pub data: Option<DeviceTensorKind>,
    pub shape: Vec<usize>,
}

impl From<DeviceBuffer<f32>> for DeviceTensorKind {
    fn from(who: DeviceBuffer<f32>) -> Self {
        DeviceTensorKind::FloatTensor(who)
    }
}

impl From<DeviceBuffer<i32>> for DeviceTensorKind {
    fn from(who: DeviceBuffer<i32>) -> Self {
        DeviceTensorKind::Int32Tensor(who)
    }
}

impl From<DeviceBuffer<f64>> for DeviceTensorKind {
    fn from(who: DeviceBuffer<f64>) -> Self {
        DeviceTensorKind::DoubleTensor(who)
    }
}

impl From<DeviceBuffer<i8>> for DeviceTensorKind {
    fn from(who: DeviceBuffer<i8>) -> Self {
        DeviceTensorKind::Int8Tensor(who)
    }
}

impl DeviceTensor {
    pub fn ndims(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    pub fn from_vec(raw_data: Vec<f32>) -> DeviceResult<DeviceTensor> {
        unsafe {
            match &G_KERNEL.1 {Some(context) => {CurrentContext::set_current(context).unwrap()} _=> {}}
        }
        
        let raw_shape = vec![raw_data.len()];
        let data = DeviceBuffer::from_slice(&raw_data);
        
        match data {
            Ok(buf) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(buf)),
                    shape: raw_shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(_e) => { println!("Failed to alloc device memory!"); Err(_e) }
        }

    }

    pub fn from_vec_shape(raw_data: Vec<f32>, shape: Vec<usize>) -> DeviceResult<DeviceTensor> {
        unsafe {
            match &G_KERNEL.1 {Some(context) => {CurrentContext::set_current(context).unwrap()} _=> {}}
        }
        let data = DeviceBuffer::from_slice(&raw_data);
        match data {
            Ok(buf) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(buf)),
                    shape: shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(_e) => { println!("Failed to alloc device memory!"); Err(_e) }
        }
    }

    pub fn from_vec_shape_i32(raw_data: Vec<i32>, shape: Vec<usize>) -> DeviceResult<DeviceTensor> {
        unsafe {
            match &G_KERNEL.1 {Some(context) => {CurrentContext::set_current(context).unwrap()} _=> {}}
        }
        let data = DeviceBuffer::from_slice(&raw_data);
        match data {
            Ok(buf) => {
                Ok(DeviceTensor {
                    data: Some(DeviceTensorKind::from(buf)),
                    shape: shape,
                })
            }
            #[cfg(test)]
            Err(_e) => { panic!("Failed to alloc device memory!"); }
            #[cfg(not(test))]
            Err(_e) => { println!("Failed to alloc device memory!"); Err(_e) }
        }
    }

    pub fn fill(shape: Vec<usize>, v : f32) -> DeviceResult<DeviceTensor> {
        let ret: usize = shape.iter().fold(1usize, |mut ret, val| {ret *= *val; ret});
        // let ret: u32 = shape.iter().product();
        Self::from_vec_shape(vec![v; ret], shape)
    }

    pub fn zeros(shape: Vec<usize>) -> DeviceResult<DeviceTensor> {
        let ret: usize = shape.iter().fold(1usize, |mut ret, val| {ret *= *val; ret});
        // let ret: u32 = shape.iter().product();
        Self::from_vec_shape(vec![0.0f32; ret], shape)
    }

    pub fn ones(shape: Vec<usize>) -> DeviceResult<DeviceTensor> {
        let ret: usize = shape.iter().fold(1usize, |mut ret, val| {ret *= *val; ret});
        // let ret: u32 = shape.iter().product();
        Self::from_vec_shape(vec![1.0f32; ret], shape)
    }

}

impl PartialEq for DeviceTensor {
    fn eq(&self, other: &Self) -> bool {
        match &self.data {
            Some(data1) => {
                match &other.data {
                    Some(data2) => {
                        let size1: usize = self.shape.iter().fold(1usize, |mut ret, val| {ret *= *val; ret});
                        let size2: usize = other.shape.iter().fold(1usize, |mut ret, val| {ret *= *val; ret});
                        if self.shape != other.shape { return false }
                        match data1 {
                            DeviceTensorKind::FloatTensor(ret1) => {
                                let mut out1 = vec![0.0f32; size1];
                                ret1.copy_to(&mut out1[0..size1]).unwrap();
                                match data2 {
                                    DeviceTensorKind::FloatTensor(ret2) => {
                                        let mut out2 = vec![0.0f32; size2];
                                        ret2.copy_to(&mut out2[0..size2]).unwrap();
                                        return float_eq!(out1, out2, rmax_all<=0.000001f32);
                                    }
                                    _ => { return false }
                                }
                            }
                            DeviceTensorKind::DoubleTensor(ret1) => {
                                let mut out1 = vec![0.0f64; size1];
                                ret1.copy_to(&mut out1[0..size1]).unwrap();
                                match data2 {
                                    DeviceTensorKind::DoubleTensor(ret2) => {
                                        let mut out2 = vec![0.0f64; size2];
                                        ret2.copy_to(&mut out2[0..size2]).unwrap();
                                        return out1 == out2
                                    }
                                    _ => { return false }
                                }
                            }
                            DeviceTensorKind::Int32Tensor(ret1) => {
                                let mut out1 = vec![0i32; size1];
                                ret1.copy_to(&mut out1[0..size1]).unwrap();
                                match data2 {
                                    DeviceTensorKind::Int32Tensor(ret2) => {
                                        let mut out2 = vec![0i32; size2];
                                        ret2.copy_to(&mut out2[0..size2]).unwrap();
                                        return out1 == out2
                                    }
                                    _ => { return false }
                                }
                            }
                            DeviceTensorKind::Int8Tensor(ret1) => {
                                let mut out1 = vec![0i8; size1];
                                ret1.copy_to(&mut out1[0..size1]).unwrap();
                                match data2 {
                                    DeviceTensorKind::Int8Tensor(ret2) => {
                                        let mut out2 = vec![0i8; size2];
                                        ret2.copy_to(&mut out2[0..size2]).unwrap();
                                        return out1 == out2
                                    }
                                    _ => { return false }
                                }
                            }
                        }

                    }
                    _ => { return false }
                }
            }
            _ => {return false}
        }

    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_zeros_1d() {
        let tensor = DeviceTensor::zeros(vec![64]);
        let data = DeviceBuffer::from_slice(&[0.0f32; 64]).unwrap();

        match tensor {
            Ok(buf) => {
                let reft = DeviceTensor {
                    data: Some(DeviceTensorKind::FloatTensor(data)),
                    shape: vec![64],
                };
                assert_eq!(buf, reft);
            }
            _ => { panic!("Failed to alloc device memory!"); }
        }   
    }

    #[test]
    fn test_zeros_2d() {
        let tensor = DeviceTensor::zeros(vec![64, 32]);
        let data = DeviceBuffer::from_slice(&[0.0f32; 64*32]).unwrap();

        match tensor {
            Ok(buf) => {
                let reft = DeviceTensor {
                    data: Some(DeviceTensorKind::FloatTensor(data)),
                    shape: vec![64, 32],
                };
                assert_eq!(buf, reft);
            }
            _ => { panic!("Failed to alloc device memory!"); }
        }   
    }

    #[test]
    fn test_ones_1d() {
        let tensor = DeviceTensor::ones(vec![64]);
        let data = DeviceBuffer::from_slice(&[1.0f32; 64]).unwrap();

        match tensor {
            Ok(buf) => {
                let reft = DeviceTensor {
                    data: Some(DeviceTensorKind::FloatTensor(data)),
                    shape: vec![64],
                };
                assert_eq!(buf, reft);
            }
            _ => { panic!("Failed to alloc device memory!"); }
        }   
    }

    #[test]
    fn test_ones_2d() {
        let tensor = DeviceTensor::ones(vec![64, 32]);
        let data = DeviceBuffer::from_slice(&[1.0f32; 64*32]).unwrap();

        match tensor {
            Ok(buf) => {
                let reft = DeviceTensor {
                    data: Some(DeviceTensorKind::FloatTensor(data)),
                    shape: vec![64, 32],
                };
                assert_eq!(buf, reft);
            }
            _ => { panic!("Failed to alloc device memory!"); }
        } 
    }
}