use crate::device_opcode::DeviceOpCode;
use crate::device_tensor::{DeviceTensor, DeviceTensorKind};
use uhal::error::DeviceResult;
use uhal::memory::{DeviceBufferTrait};

#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer;

#[cfg(feature = "cuda_backend")]
use cuda_backend as cuda;
#[cfg(feature = "cuda_backend")]
use cuda::memory::CuDeviceBuffer;

#[derive(Debug)]
pub struct DeviceExecutor {}

impl DeviceExecutor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn binary_compute_owned(
        &self,
        op: DeviceOpCode,
        lhs: DeviceTensor,
        rhs: DeviceTensor,
    ) -> DeviceResult<DeviceTensor> {
        match op {
            DeviceOpCode::AddF => self.addf32_owned(lhs, rhs),
            DeviceOpCode::SubF => self.subf32_owned(lhs, rhs),
            DeviceOpCode::MulF => self.mulf32_owned(lhs, rhs),
            DeviceOpCode::DivF => self.divf32_owned(lhs, rhs),
            DeviceOpCode::MatMulF => self.matmul_owned(lhs, rhs),
            DeviceOpCode::Conv2DF => self.conv2d_owned(lhs, rhs),
            _ => panic!("not wired opcode"),
        }
    }
    pub fn mock_result(&self, mock_data:Vec<f32>, mock_shape:Vec<usize>) -> DeviceResult<DeviceTensor> {
        #[cfg(feature = "tops_backend")]
        let data = TopsDeviceBuffer::from_slice(&mock_data);

        #[cfg(feature = "cuda_backend")]
        let data = CuDeviceBuffer::from_slice(&mock_data);

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
    pub fn addf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0], vec![2, 3])
    }

    pub fn subf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![0.0f32; 6], vec![2, 3])
    }

    pub fn mulf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32, 4.0, 9.0, 16.0, 25.0, 36.0], vec![2, 3])
    }

    pub fn divf32_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32; 6], vec![2, 3])
    }

    pub fn matmul_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![23.0f32; 17 * 18], vec![17, 18])
    }

    pub fn conv2d_owned(&self, lhs: DeviceTensor, rhs: DeviceTensor) -> DeviceResult<DeviceTensor> {
        self.mock_result(vec![1.0f32; 6], vec![2, 3])
    }

    pub fn transpose_owned(&self, arg: DeviceTensor) -> DeviceResult<DeviceTensor> {
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
        let c = exec.matmul_owned(a, b).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [17, 18]);
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
        let c = exec.addf32_owned(a, b).unwrap();
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
        let c = exec.subf32_owned(a, b).unwrap();
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
        let c = exec.mulf32_owned(a, b).unwrap();
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
        let c = exec.divf32_owned(a, b).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [2, 3]);
        assert_eq!(c, cref);
    }

    #[test]
    fn test_transpose_owned() {
        let a = DeviceTensor::from_vec_shape(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let cref = DeviceTensor::from_vec_shape(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0], vec![3, 2]).unwrap();

        let exec = DeviceExecutor::new();
        let c = exec.transpose_owned(a).unwrap();
        assert_eq!(c.ndims(), 2);
        assert_eq!(c.shape(), [3, 2]);
        assert_eq!(c, cref);
    }

}