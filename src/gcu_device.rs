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
*
* @file    gcu_device.rs
* @brief
*
* @author  Guoqing Bao
* @date    2023-09-05 - 2024-01-10
* @version V0.1
* @par     Copyright (c) Enflame Tech Company.
* @par     History: gemm tuner cache
* @par     Comments: a gcu device abstraction facilitating memory allocation, memcpy, and stream syncronization.
*/
use std::ffi::c_void;

use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};
use tops::stream::topsStream_t;
use uhal::error::{DeviceError, DeviceResult};

use crate::gemm_tuner::{AtenGemmInfo, AtenGemmTuner, GEMM_OP_PARAS};
use crate::prelude::GcuLaunchConfig;
pub use cust_core::_hidden::DeviceCopy;
use std::string::String;
use tops::device::TopsDevice as Device;
use tops::stream::TopsStream as Stream;
use uhal::memory::{DeviceBufferTrait, DevicePointerTrait};
//Tops backend
use crate::device_executor::DeviceExecutor;
use crate::device_ptr::DevicePtr;
use crate::gcu_slice::GcuSlice;
use driv::topsFunction_t;
use lazy_static::lazy_static;
use std::sync::Mutex;
pub use tops::driv;
use tops::error::ToResult;
use tops::memory::CopyDestination;
use tops::memory::TopsDeviceBuffer as DeviceBuffer;
use tops::TopsApi as Api;
use tops_backend as tops;
use uhal::device::DeviceTrait;
use uhal::stream::{StreamFlags, StreamTrait};
use uhal::DriverLibraryTrait;

// Define a global Mutex
lazy_static! {
    static ref GLOBAL_LOCK: Mutex<()> = Mutex::new(());
}
// #[derive(Clone)]
pub struct GcuDevice {
    pub id: usize,
    pub executor: Arc<DeviceExecutor>,
    is_async: bool,
    prop: driv::topsDeviceProp_t,
    pub launch_cfg: GcuLaunchConfig,
    pub tuner: AtenGemmTuner,
    pub device: Option<Device>,
    pub stream: Option<Stream>,
}

pub struct GcuFunction {
    pub func_name: String,
    pub module_name: String,
    pub func: Option<topsFunction_t>,
    pub stream: Option<topsStream_t>,
    pub is_async: bool,
    // pub executor: Option<Arc<Mutex<&'static mut DeviceExecutor>>>,
}

impl GcuFunction {
    pub fn new(func_name: String, module_name: String) -> Self {
        GcuFunction {
            func_name,
            module_name,
            func: None,
            stream: None,
            is_async: false,
        }
    }
}

impl std::fmt::Debug for GcuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcuDevice({:?})", self.id)
    }
}

impl std::ops::Deref for GcuDevice {
    type Target = Option<Device>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl GcuDevice {
    pub fn new(ordinal: usize, eager_mode: bool) -> DeviceResult<Arc<Self>> {
        let (device, stream) = match Api::quick_init(ordinal as u32) {
            Ok(device) => match Stream::new(StreamFlags::NON_BLOCKING, None) {
                Ok(stream) => (device, stream),
                _ => panic!("Failed to create stream!"),
            },
            _ => panic!("Failed to create device!"),
        };
        let gcu_executor = DeviceExecutor::new(ordinal as u32);
        let mut prop = driv::topsDeviceProp_t::default();
        unsafe {
            driv::topsGetDeviceProperties(&mut prop as *mut driv::topsDeviceProp_t, ordinal as i32);
        }
        let mut property = format!("**GCU device property: {:?} \n", prop);
        property = property.replace(" 0, 0,", "");
        println!("{}", property);
        Ok(Arc::new(Self {
            id: ordinal,
            executor: Arc::new(gcu_executor),
            is_async: !eager_mode,
            prop,
            launch_cfg: GcuLaunchConfig {
                grid_dim: (prop.multiProcessorCount as u32, 1, 1),
                block_dim: (prop.maxThreadsPerMultiProcessor as u32, 1, 1),
                shared_mem_bytes: 0,
                is_cooperative_launch: false,
            },
            tuner: AtenGemmTuner::new(),
            device: Some(device),
            stream: Some(stream),
        }))
    }

    //TODO: Fix this with topsCtxSetCurrent
    //Given that device context is not support at the moment on GCU,
    //we temporarily use device select (which is not stable) instead of topsCtxSetCurrent
    pub fn bind_to_thread(self: &Arc<Self>) -> DeviceResult<()> {
        let _guard = GLOBAL_LOCK.lock().unwrap(); // Acquire the lock
        Device::select_device(self.ordinal() as u32)
    }

    pub fn stream_inner(&self) -> Option<topsStream_t> {
        if self.stream.is_some() {
            Some(self.stream.as_ref().unwrap().as_inner() as topsStream_t)
        } else {
            None
        }
    }

    pub fn ordinal(&self) -> usize {
        self.id
    }
    pub fn get_or_load_func(
        &self,
        func_name: &str,
        module_name: &str,
    ) -> DeviceResult<GcuFunction> {
        if self
            .executor
            .has_function(module_name.to_string(), func_name.to_string())
        {
            return Ok(GcuFunction {
                func_name: func_name.to_string(),
                module_name: module_name.to_string(),
                func: Some(self.executor.function_map[func_name].as_inner()),
                stream: self.stream_inner(),
                is_async: self.is_async,
            });
        } else {
            println!("Kernel {} not found!", func_name);
        }
        Ok(GcuFunction::new(
            func_name.to_string(),
            module_name.to_string(),
        ))
    }

    pub fn get_gemm_launch_params(
        &self,
        datatype: crate::DATATYPE,
        weight_type: crate::DATATYPE,
        b: usize,
        m: usize,
        k: usize,
        n: usize,
        rhs_trans: i32,
    ) -> &GEMM_OP_PARAS {
        // let bias = self.alloc::<f16>(n).w()?;
        let info = AtenGemmInfo::new(datatype, weight_type, b, m, k, n, rhs_trans);
        unsafe { self.tuner.tuner(&info) }
    }
    /// Allocates device memory and increments the reference counter of [GcuDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub fn alloc<T: DeviceCopy>(self: &Arc<Self>, len: usize) -> DeviceResult<GcuSlice<T>> {
        let device_ptr = if self.is_async {
            // println!("alloc async!  (len={})", len);
            unsafe { DeviceBuffer::uninitialized_async(len, self.stream.as_ref().unwrap())? }
        } else {
            unsafe { DeviceBuffer::uninitialized(len)? }
        };
        Ok(GcuSlice {
            buffer: device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
            host_buf_ptr: None,
        })
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros<T: DeviceCopy>(self: &Arc<Self>, len: usize) -> DeviceResult<GcuSlice<T>> {
        let device_ptr = if self.is_async {
            unsafe {
                // println!("alloc_zeros async! (len={})", len);
                let device_ptr =
                    DeviceBuffer::uninitialized_async(len, self.stream.as_ref().unwrap())?;
                driv::topsMemsetD8Async(
                    device_ptr.as_device_ptr().as_raw(),
                    0,
                    (std::mem::size_of::<T>() * len) as u64,
                    self.stream_inner().expect("unable to obtain stream"),
                )
                .to_result()?;
                device_ptr
            }
        } else {
            unsafe {
                let device_ptr = DeviceBuffer::uninitialized(len)?;
                driv::topsMemsetD8(
                    device_ptr.as_device_ptr().as_raw(),
                    0,
                    (std::mem::size_of::<T>() * len) as u64,
                )
                .to_result()?;
                device_ptr
            }
        };

        // unsafe {
        //     driv::topsMemsetD8(device_ptr.as_device_ptr().as_raw(), 0, std::mem::size_of::<T>() * len).to_result()?;
        // }

        Ok(GcuSlice {
            buffer: device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
            host_buf_ptr: None,
        })
    }

    /// Sets all memory to 0 asynchronously.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    // pub fn memset_zeros<T: DeviceCopy>(
    //     self: &Arc<Self>,
    //     dst: &mut Dst,
    // ) -> DeviceResult<()> {
    //     self.bind_to_thread()?;
    //     if self.is_async {
    //         unsafe {
    //             result::memset_d8_async(*dst.device_ptr_mut(), 0, dst.num_bytes(), self.stream)
    //         }
    //     } else {
    //         unsafe { result::memset_d8_sync(*dst.device_ptr_mut(), 0, dst.num_bytes()) }
    //     }
    // }

    /// Device to device copy (safe version of [result::memcpy_dtod_async]).
    ///
    /// # Panics
    ///
    /// If the length of the two values are different
    ///
    /// # Safety
    /// 1. We are guarunteed that `src` and `dst` are pointers to the same underlying
    ///     type `T`
    /// 2. Since they are both references, they can't have been freed
    /// 3. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtod_copy<T: DeviceCopy, Src: DevicePtr<T>, Dst: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut Dst,
    ) -> DeviceResult<()> {
        assert!(src.len() <= dst.len());
        if self.is_async {
            // println!("dtod_copy async! (len={}, len={})", src.len(), dst.len());
            unsafe {
                driv::topsMemcpyDtoDAsync(
                    dst.device_ptr(),
                    src.device_ptr(),
                    (src.len() * std::mem::size_of::<T>()) as u64,
                    self.stream_inner().expect("unable to obtain stream"),
                )
                .to_result()
            }
        } else {
            unsafe {
                driv::topsMemcpyDtoD(
                    dst.device_ptr(),
                    src.device_ptr(),
                    (src.len() * std::mem::size_of::<T>()) as u64,
                )
                .to_result()
            }
        }
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy<T: DeviceCopy + Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> DeviceResult<GcuSlice<T>> {
        let mut dst = self.alloc(src.len())?;
        self.htod_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    pub unsafe fn memcpy_htod_async<T>(
        dst: driv::topsDeviceptr_t,
        src: &[T],
        stream: driv::topsStream_t,
    ) -> DeviceResult<()> {
        let size = std::mem::size_of_val(src) as u64;
        let ptr = src.as_ptr() as *mut _;
        driv::topsHostRegister(ptr, size, 0).to_result()?;
        driv::topsMemcpyHtoDAsync(dst, ptr, size, stream).to_result()
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy_into<T: DeviceCopy + Unpin>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut GcuSlice<T>,
    ) -> DeviceResult<()> {
        assert_eq!(src.len(), dst.len);
        let size = std::mem::size_of::<T>() * src.len();

        // let host_ptr = dst.host_buf.as_ref().unwrap().as_ptr() as *mut c_void;
        if self.is_async {
            unsafe {
                // if size % 256 != 0 {
                let mut ptr = std::ptr::null_mut();
                driv::topsHostMalloc(&mut ptr as *mut *mut c_void, size as u64, 0).to_result()?;
                std::ptr::copy(src.as_ptr() as *mut c_void, ptr, size);
                dst.host_buf_ptr = Some(ptr);
                return driv::topsMemcpyHtoDAsync(
                    dst.device_ptr(),
                    ptr,
                    size as u64,
                    self.stream_inner().expect("unable to obtain stream"),
                )
                .to_result();
                // } else {
                //     dst.host_buf = Some(Pin::new(src));
                //     return Self::memcpy_htod_async(dst.device_ptr(), dst.host_buf.as_ref().unwrap(), self.stream.unwrap().as_inner());
                // }
            };
        } else {
            dst.buffer.copy_from(&src)
        }
    }

    /// Allocates new device memory and synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [GcuDevice::htod_copy()].
    ///
    /// # Safety
    ///
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy<T: DeviceCopy>(self: &Arc<Self>, src: &[T]) -> DeviceResult<GcuSlice<T>> {
        let device_ptr = unsafe { DeviceBuffer::uninitialized(src.len())? };
        let mut dst = GcuSlice {
            buffer: device_ptr,
            len: src.len(),
            device: self.clone(),
            host_buf: None,
            host_buf_ptr: None,
        };

        dst.buffer.copy_from(src)?;
        Ok(dst)
    }

    /// Synchronously copies data from `src` into the new allocation.
    ///
    /// If you want an asynchronous copy, see [GcuDevice::htod_copy()].
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `src` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_sync_copy_into<
        T: DeviceCopy,
        I: AsRef<[T]> + AsMut<[T]> + ?Sized,
        Dst: DevicePtr<T>,
    >(
        self: &Arc<Self>,
        src: &I,
        dst: &mut Dst,
    ) -> DeviceResult<()> {
        let val = src.as_ref();
        assert_eq!(val.len(), dst.len());
        if self.is_async {
            unsafe {
                driv::topsMemcpyHtoDAsync(
                    dst.device_ptr(),
                    val.as_ptr() as *mut c_void,
                    std::mem::size_of_val(val) as u64,
                    self.stream_inner().expect("unable to obtain stream"),
                )
                .to_result()?;
            }
        } else {
            unsafe {
                driv::topsMemcpyHtoD(
                    dst.device_ptr(),
                    val.as_ptr() as *mut c_void,
                    std::mem::size_of_val(val) as u64,
                )
                .to_result()?;
            }
        }
        self.synchronize()
    }

    /// Synchronously copies device memory into host memory.
    /// Unlike [`GcuDevice::dtoh_sync_copy_into`] this returns a [`Vec<T>`].
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` (after returning) it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    #[allow(clippy::uninit_vec)]
    pub fn dtoh_sync_copy<T: DeviceCopy>(
        self: &Arc<Self>,
        src: &GcuSlice<T>,
    ) -> DeviceResult<Vec<T>> {
        let mut dst = Vec::with_capacity(src.len);
        unsafe { dst.set_len(src.len) };
        self.dtoh_sync_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Synchronously copies device memory into host memory
    ///
    /// Use [`GcuDevice::dtoh_sync_copy`] if you need [`Vec<T>`] and can't provide
    /// a correctly sized slice.
    ///
    /// # Panics
    ///
    /// If the lengths of slices are not equal, this method panics.
    ///
    /// # Safety
    /// 1. Since this function doesn't own `dst` it is executed synchronously.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn dtoh_sync_copy_into<T: DeviceCopy, Src: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &Src,
        dst: &mut [T],
    ) -> DeviceResult<()> {
        assert_eq!(src.len(), dst.len());
        let val = dst;
        if self.is_async {
            unsafe {
                driv::topsMemcpyDtoHAsync(
                    val.as_mut_ptr() as *mut c_void,
                    src.device_ptr(),
                    std::mem::size_of_val(val) as u64,
                    self.stream_inner().expect("unable to obtain stream"),
                )
                .to_result()?;
            }
        } else {
            unsafe {
                driv::topsMemcpyDtoH(
                    val.as_mut_ptr() as *mut c_void,
                    src.device_ptr(),
                    std::mem::size_of_val(val) as u64,
                )
                .to_result()?;
            }
        }
        self.synchronize()
    }

    /// Synchronously de-allocates `src` and converts it into it's host value.
    /// You can just [drop] the slice if you don't need the host data.
    ///
    /// # Safety
    /// 1. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn sync_reclaim<T: Clone + Default + DeviceCopy + Unpin>(
        self: &Arc<Self>,
        mut src: GcuSlice<T>,
    ) -> DeviceResult<Vec<T>> {
        let buf = src.host_buf.take();
        let mut buf = buf.unwrap_or_else(|| {
            let mut b = Vec::with_capacity(src.len);
            b.resize(src.len, Default::default());
            Pin::new(b)
        });
        self.dtoh_sync_copy_into(&src, &mut buf)?;
        Ok(Pin::into_inner(buf))
    }

    /// Synchronizes the stream.
    pub fn synchronize(self: &Arc<Self>) -> DeviceResult<()> {
        match self.stream.as_ref() {
            Some(_stream) => _stream.synchronize(),
            _ => {
                println!("Unable to use stream!");
                Ok(())
            }
        }
    }

    pub fn get_block_number(&self) -> i32 {
        self.prop.multiProcessorCount
    }

    pub fn get_thread_number(&self) -> i32 {
        // if self.prop.gcuArchName.join("") == "dtu-enflame-tops--gcu210" {
        //     let mut block_num = 0i32;
        //     unsafe {
        //         driv::topsDeviceGetAttribute(&mut block_num as *mut i32, driv::topsDeviceAttribute_t::topsDeviceAttributeMaxThreadsPerBlock, self.id as i32);
        //     }
        //   return block_num;
        // }

        self.prop.maxThreadsPerBlock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcu_slice::RangeHelper;
    #[test]
    fn test_post_build_arc_count() {
        let device = GcuDevice::new(0, true).unwrap();
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_alloc_arc_counts() {
        let device = GcuDevice::new(0, true).unwrap();
        let t = device.alloc_zeros::<f32>(1).unwrap();
        assert!(t.host_buf.is_none());
        assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    fn test_post_take_arc_counts() {
        let device = GcuDevice::new(0, true).unwrap();
        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        assert!(t.host_buf.is_some());
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_counts() {
        let device = GcuDevice::new(0, true).unwrap();
        let t = device.htod_copy([0.0f64; 10].to_vec()).unwrap();
        // let r = t.clone();
        // assert_eq!(Arc::strong_count(&device), 3);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 1);
        // drop(r);
        // assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_clone_arc_slice_counts() {
        let device = GcuDevice::new(0, true).unwrap();
        let t = Arc::new(device.htod_copy::<f64>([0.0; 10].to_vec()).unwrap());
        let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);
        drop(t);
        assert_eq!(Arc::strong_count(&device), 2);
        drop(r);
        assert_eq!(Arc::strong_count(&device), 1);
    }

    #[test]
    fn test_post_release_counts() {
        let device = GcuDevice::new(0, true).unwrap();
        let t = device.htod_copy([1.0f32, 2.0, 3.0].to_vec()).unwrap();
        // let r = t.clone();
        assert_eq!(Arc::strong_count(&device), 2);

        let r_host = device.sync_reclaim(t).unwrap();
        assert_eq!(&r_host, &[1.0, 2.0, 3.0]);
        assert_eq!(Arc::strong_count(&device), 1);

        drop(r_host);
        // assert_eq!(Arc::strong_count(&device), 2);
    }

    #[test]
    #[ignore = "must be executed by itself"]
    fn test_post_alloc_memory() {
        let device = GcuDevice::new(0, true).unwrap();
        // let (free1, total1) = result::mem_get_info().unwrap();

        let t = device.htod_copy([0.0f32; 5].to_vec()).unwrap();
        // let (free2, total2) = result::mem_get_info().unwrap();
        // assert_eq!(total1, total2);
        // assert!(free2 < free1);

        drop(t);
        device.synchronize().unwrap();

        // let (free3, total3) = result::mem_get_info().unwrap();
        // assert_eq!(total2, total3);
        // assert!(free3 > free2);
        // assert_eq!(free3, free1);
    }

    #[test]
    fn test_device_copy_to_views() {
        let dev = GcuDevice::new(0, true).unwrap();

        let smalls = [
            dev.htod_copy(std::vec![-1.0f32, -0.8]).unwrap(),
            dev.htod_copy(std::vec![-0.6, -0.4]).unwrap(),
            dev.htod_copy(std::vec![-0.2, 0.0]).unwrap(),
            dev.htod_copy(std::vec![0.2, 0.4]).unwrap(),
            dev.htod_copy(std::vec![0.6, 0.8]).unwrap(),
        ];
        let mut big = dev.alloc_zeros::<f32>(10).unwrap();

        let mut offset = 0;
        for small in smalls.iter() {
            let mut sub = big.try_slice_mut(offset..offset + small.len).unwrap();
            dev.dtod_copy(small, &mut sub).unwrap();
            offset += small.len;
        }

        assert_eq!(
            dev.sync_reclaim(big).unwrap(),
            [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
        );
    }

    #[test]
    fn test_leak_and_upgrade() {
        let dev = GcuDevice::new(0, true).unwrap();

        let _ = dev
            .htod_copy(std::vec![1.0f32, 2.0, 3.0, 4.0, 5.0])
            .unwrap();

        // let ptr = a.leak();
        // let b = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 3) };
        // assert_eq!(dev.dtoh_sync_copy(&b).unwrap(), &[1.0, 2.0, 3.0]);

        // let ptr = b.leak();
        // let c = unsafe { dev.upgrade_device_ptr::<f32>(ptr, 5) };
        // assert_eq!(dev.dtoh_sync_copy(&c).unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// See https://github.com/coreylowman/Gcurc/issues/160
    #[test]
    fn test_slice_is_freed_with_correct_context() {
        let dev0 = GcuDevice::new(0, true).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let dev1 = GcuDevice::new(1, true).unwrap();
        drop(dev1);
        drop(dev0);
        drop(slice);
    }

    /// See https://github.com/coreylowman/Gcurc/issues/161
    #[test]
    fn test_copy_uses_correct_context() {
        let dev0 = GcuDevice::new(0, true).unwrap();
        let _dev1 = GcuDevice::new(1, true).unwrap();
        let slice = dev0.htod_copy(vec![1.0; 10]).unwrap();
        let _out = dev0.dtoh_sync_copy(&slice).unwrap();
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)]
    fn test_bounds_helper() {
        assert_eq!((..2usize).bounds(0..usize::MAX), Some((0, 2)));
        assert_eq!((1..2usize).bounds(..usize::MAX), Some((1, 2)));
        assert_eq!((..).bounds(1..10), Some((1, 10)));
        assert_eq!((2..=2usize).bounds(0..usize::MAX), Some((2, 3)));
        assert_eq!((2..=2usize).bounds(0..=1), None);
        assert_eq!((2..2usize).bounds(0..usize::MAX), Some((2, 2)));
        assert_eq!((1..0usize).bounds(0..usize::MAX), None);
        assert_eq!((1..=0usize).bounds(0..usize::MAX), None);
    }

    #[test]
    fn test_transmutes() {
        let dev = GcuDevice::new(0, true).unwrap();
        let mut slice = dev.alloc_zeros::<u8>(100).unwrap();
        assert!(unsafe { slice.transmute::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute::<f32>(26) }.is_none());
        assert!(unsafe { slice.transmute_mut::<f32>(25) }.is_some());
        assert!(unsafe { slice.transmute_mut::<f32>(26) }.is_none());
    }
}
