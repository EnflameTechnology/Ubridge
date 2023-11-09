use std::ffi::c_void;

use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};
use tops::memory::AsyncCopyDestination;
use uhal::device::DeviceTrait;
use uhal::error::{DeviceResult};

use uhal::memory::{DeviceBufferTrait, DevicePointerTrait};
use uhal::module::ModuleTrait;
use uhal::stream::{StreamTrait};
use uhal::DriverLibraryTrait;
pub use cust_core::_hidden::{DeviceCopy};
use std::{
    ops::{RangeBounds},
    string::String,
};
use std::path::Path;

//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
#[cfg(feature = "tops_backend")]
use tops::device::TopsDevice as Device;
#[cfg(feature = "tops_backend")]
use tops::memory::CopyDestination;
#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;

#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;


#[cfg(feature = "tops_backend")]
pub use tops::driv as driv;
use driv::{topsFunction_t};
use tops::error::ToResult;
use crate::device_executor::DeviceExecutor;
use crate::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};
use crate::gcu_slice::{GcuSlice, RangeHelper};

#[derive(Clone)]
pub struct GcuDevice {
    pub id: usize,
    device: Option<&'static Device>,
    stream: Option<&'static Stream>,
    executor: Arc<&'static mut DeviceExecutor>,
    is_async: bool,
}

pub struct GcuFunction {
    pub name: String,
    pub path: String,
    pub func: Option<topsFunction_t>,
    // pub executor: Option<Arc<Mutex<&'static mut DeviceExecutor>>>,
}

impl GcuFunction {
    pub fn new(name: String, path: String) -> Self {
        GcuFunction {
            name: name,
            path: path,
            func: None,
            // executor: None
        }
    }

    // pub fn launch(&self) {

    // }
}

impl std::fmt::Debug for GcuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GcuDevice({:?})", self.id)
    }
}

impl std::ops::Deref for GcuDevice {
    type Target = Option<&'static Device>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl GcuDevice {
    pub fn new(ordinal: usize, eager_mode: bool) -> DeviceResult<Arc<Self>> {
        match DeviceExecutor::get_gcu_executor(ordinal as u32) {
            Some(gcu_executor) => { 
                Ok(Arc::new(Self {
                    id: ordinal,
                    device: gcu_executor.device,
                    stream: gcu_executor.stream,
                    executor: Arc::new(gcu_executor),
                    is_async: !eager_mode
                }))
            }
            _=> {
                panic!("Unable to obtain GCU device!");
            }
        }
        // let (device, stream) = match Api::quick_init(ordinal) {
        //     Ok(_device) => {
        //         match Stream::new(StreamFlags::NON_BLOCKING, None) {
        //             Ok(_stream) => (_device, _stream),
        //             _ => {
        //                 panic!("Unable to create stream!");
        //             }
        //         };
        //     }
        //     _ => {panic!("Unable to obtain GCU device!");}
        // };


    }

    pub fn ordinal(&self) -> usize {
        self.id
    }
    pub fn get_or_load_func(&self, func_name: &str, kernel_path: &str) -> DeviceResult<GcuFunction> {
        let path = Path::new(kernel_path);
        let _module_name = path.file_stem().unwrap().to_str().unwrap();
        if (_module_name == "unary" && self.executor.has_function(_module_name.to_string(), func_name.to_string())) 
        || (_module_name == "binary" && self.executor.has_function(_module_name.to_string(), func_name.to_string())) 
        || (_module_name == "affine" && self.executor.has_function(_module_name.to_string(), func_name.to_string())) 
        || _module_name=="dotllm"
        || _module_name=="batch_matmul" 
        // if _module_name=="dotllm" 
        {
            match &self.executor.function_map{
                Some(funcs) => {
                    return Ok(
                        GcuFunction {
                            name: func_name.to_string(), 
                            path: kernel_path.to_string(),
                            func: Some(funcs[func_name].inner),
                        }
                    );
                }
                _=> {}
            }

        }
        Ok(GcuFunction::new(func_name.to_string(), kernel_path.to_string())) //TODO, write kernels
    }
    /// Allocates device memory and increments the reference counter of [GcuDevice].
    ///
    /// # Safety
    /// This is unsafe because the device memory is unset after this call.
    pub fn alloc<T: DeviceCopy>(
        self: &Arc<Self>,
        len: usize,
    ) -> DeviceResult<GcuSlice<T>> {
        // self.bind_to_thread()?;
        let device_ptr = if self.is_async {
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream.unwrap())? }
        } else {
            unsafe { DeviceBuffer::uninitialized(len)? }
        };
        Ok(GcuSlice {
            buffer: device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
        })
    }

    /// Allocates device memory with no associated host memory, and memsets
    /// the device memory to all 0s.
    ///
    /// # Safety
    /// 1. `T` is marked as [ValidAsZeroBits], so the device memory is valid to use
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn alloc_zeros<T: DeviceCopy>(
        self: &Arc<Self>,
        len: usize,
    ) -> DeviceResult<GcuSlice<T>> {

        let device_ptr = if self.is_async {
            unsafe { DeviceBuffer::uninitialized_async(len, &self.stream.unwrap())? }
        } else {
            unsafe { DeviceBuffer::uninitialized(len)? }
        };

        unsafe {
            driv::topsMemsetD8(device_ptr.as_device_ptr().as_raw(), 0, std::mem::size_of::<T>() * len).to_result()?;
        }

        Ok(GcuSlice {
            buffer: device_ptr,
            len,
            device: self.clone(),
            host_buf: None,
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
        assert_eq!(src.len(), dst.len());
        // self.bind_to_thread()?;
        if self.is_async {
            unsafe { driv::topsMemcpyDtoDAsync(
                dst.device_ptr().clone(),
                src.device_ptr().clone(),
                src.len() * std::mem::size_of::<T>(),
                self.stream.unwrap().as_inner(),
            ).to_result() }

            // unsafe { src.device_ptr() async_copy_to(&mut dst.buffer, &self.stream.unwrap()) }
        } else {
            // src.buffer.copy_to(&mut dst.buffer)

            unsafe { driv::topsMemcpyDtoD(
                dst.device_ptr().clone(),
                src.device_ptr().clone(),
                src.len() * std::mem::size_of::<T>(),
            ).to_result() }
        }
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy<T: DeviceCopy>(
        self: &Arc<Self>,
        src: Vec<T>,
    ) -> DeviceResult<GcuSlice<T>> {
        let mut dst = self.alloc(src.len())?;
        self.htod_copy_into(src, &mut dst)?;
        Ok(dst)
    }

    /// Takes ownership of the host data and copies it to device data asynchronously.
    ///
    /// # Safety
    ///
    /// 1. Since `src` is owned by this funcion, it is safe to copy data. Any actions executed
    ///    after this will take place after the data has been successfully copied.
    /// 2. Self is [`Arc<Self>`], and this method increments the rc for self
    pub fn htod_copy_into<T: DeviceCopy>(
        self: &Arc<Self>,
        src: Vec<T>,
        dst: &mut GcuSlice<T>,
    ) -> DeviceResult<()> {
        assert_eq!(src.len(), dst.len);
        // dst.host_buf = Pin::new(src);
        // self.bind_to_thread()?;
        if self.is_async {
            return unsafe {
                dst.buffer.async_copy_from(
                &src,
                &self.stream.unwrap(),
            )
            };
        } else {
            return dst.buffer.copy_from(
                &src,
            );
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
    pub fn htod_sync_copy<T: DeviceCopy>(
        self: &Arc<Self>,
        src: &[T],
    ) -> DeviceResult<GcuSlice<T>> {
        let mut dst = self.alloc(src.len())?;
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
    pub fn htod_sync_copy_into<T: DeviceCopy, I: AsRef<[T]> + AsMut<[T]> + ?Sized, Dst: DevicePtr<T>>(
        self: &Arc<Self>,
        src: &I,
        dst: &mut Dst,
    ) -> DeviceResult<()> {
        let val = src.as_ref();
        assert_eq!(val.len(), dst.len());
        // self.bind_to_thread()?;
        if self.is_async {
            unsafe { 
               driv::topsMemcpyHtoDAsync(
                dst.device_ptr().clone(),
                val.as_ptr() as *mut c_void,
                val.len() * std::mem::size_of::<T>(),
                self.stream.unwrap().as_inner(),
                ).to_result()?
            }

            // unsafe {dst.buffer.async_copy_from(src, &self.stream.unwrap())?;}
            self.synchronize()
        } else {
            // dst.buffer.copy_from(src)
            unsafe { 
                driv::topsMemcpyHtoD(
                 dst.device_ptr().clone(),
                 val.as_ptr() as *mut c_void,
                 val.len() * std::mem::size_of::<T>(),
                 ).to_result()
             }
        }
        
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
        // self.bind_to_thread()?;
        let val = dst.as_mut();
        if self.is_async {
            // unsafe { src.buffer.async_copy_to(dst, &self.stream.unwrap())?; }
            unsafe { 
                driv::topsMemcpyDtoHAsync(
                    val.as_mut_ptr() as *mut c_void,
                    src.device_ptr().clone(),
                    val.len() * std::mem::size_of::<T>(),
                    self.stream.unwrap().as_inner(),
                 ).to_result()?
             }
            self.synchronize()
        } else {
            // src.buffer.copy_to(dst)
            unsafe { 
                driv::topsMemcpyDtoH(
                    val.as_mut_ptr() as *mut c_void,
                    src.device_ptr().clone(),
                    val.len() * std::mem::size_of::<T>(),
                 ).to_result()
             }
        }
        
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
        // self.bind_to_thread()?;
        match self.stream {
            Some(_stream) => { _stream.synchronize() }
            _=> {panic!("Unable to use stream!")}
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

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
        #[allow(clippy::redundant_clone)]
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

        let a = dev
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



