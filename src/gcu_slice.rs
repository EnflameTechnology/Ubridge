use std::ffi::c_void;
use std::{marker::Unpin, pin::Pin, sync::Arc, vec::Vec};
use uhal::error::{DeviceResult, DeviceError};
use uhal::memory::{DeviceBufferTrait, DevicePointerTrait};
use std::{
    marker::PhantomData,
    ops::{Bound, RangeBounds},
};
use crate::gcu_device::GcuDevice;
pub use cust_core::_hidden::{DeviceCopy};

//Tops backend
#[cfg(feature = "tops_backend")]
use tops_backend as tops;


#[cfg(feature = "tops_backend")]
use tops::memory::TopsDeviceBuffer as DeviceBuffer;

#[cfg(feature = "tops_backend")]
pub use tops::driv as driv;

use crate::device_ptr::{DevicePtr, DevicePtrMut, DeviceSlice};

#[derive(Debug)]
pub struct GcuSlice<T: DeviceCopy> {
    pub buffer: DeviceBuffer<T>,
    pub len : usize,
    pub device: Arc<GcuDevice>,
    pub host_buf: Option<Pin<Vec<T>>>,
}

unsafe impl<T: Send + DeviceCopy> Send for GcuSlice<T> {}
unsafe impl<T: Sync + DeviceCopy> Sync for GcuSlice<T> {}


impl<T: DeviceCopy> GcuSlice<T> {
    /// Get a clone of the underlying [GcuDevice].
    pub fn device(&self) -> Arc<GcuDevice> {
        self.device.clone()
    }
}

impl<T: DeviceCopy> GcuSlice<T> {
    /// Allocates copy of self and schedules a device to device copy of memory.
    pub fn try_clone(&self) -> DeviceResult<Self> {
        let mut dst = unsafe { self.device.alloc(self.len) }?;
        self.device.dtod_copy(self, &mut dst)?;
        Ok(dst)
    }
}

impl<T: DeviceCopy> Clone for GcuSlice<T> {
    fn clone(&self) -> Self {
        self.try_clone().unwrap()
    }
}

impl<T: Clone + Default + DeviceCopy + Unpin> TryFrom<GcuSlice<T>> for Vec<T> {
    type Error = DeviceError;
    fn try_from(value: GcuSlice<T>) -> DeviceResult<Self> {
        value.device.clone().sync_reclaim(value)
    }
}

/// A immutable sub-view into a [GcuSlice] created by [GcuSlice::try_slice()].
#[derive(Debug)]
pub struct GcuView<'a, T> {
    pub(crate) root: &'a driv::topsDeviceptr_t,
    pub(crate) ptr: driv::topsDeviceptr_t,
    pub(crate) len: usize,
    marker: PhantomData<T>,
}

impl<T: DeviceCopy> GcuSlice<T> {
    /// Creates a [GcuView] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `range.start >= self.len`
    pub fn slice(&self, range: impl RangeBounds<usize>) -> GcuView<'_, T> {
        self.try_slice(range).unwrap()
    }

    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    /// Fallible version of [CudaSlice::slice]
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<GcuView<'_, T>> {
        
        range.bounds(..self.len).map(|(start, end)| GcuView {
            root: self.buffer.as_device_ptr_ref().as_ref(),
            // ptr: unsafe { self.buffer.as_device_ptr().offset(isize::try_from(start * std::mem::size_of::<T>()).unwrap()).as_raw()},
            ptr: unsafe { self.buffer.as_device_ptr().offset(isize::try_from(start).unwrap()).as_raw()},
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute<S>(&self, len: usize) -> Option<GcuView<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(GcuView {
            root: self.buffer.as_device_ptr_ref().as_ref(),
            ptr: self.buffer.as_device_ptr().as_raw(),
            len,
            marker: PhantomData,
        })
    }
}

impl<'a, T> GcuView<'a, T> {
    /// Creates a [GcuView] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `range.start >= self.len`
    pub fn slice(&self, range: impl RangeBounds<usize>) -> GcuView<'a, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [GcuView::slice]
    pub fn try_slice(&self, range: impl RangeBounds<usize>) -> Option<GcuView<'a, T>> {
        range.bounds(..self.len).map(|(start, end)| GcuView {
            root: self.root,
            ptr: (self.ptr as u64 + (start * std::mem::size_of::<T>()) as u64) as *mut c_void,
            len: end - start,
            marker: PhantomData,
        })
    }
}

/// A mutable sub-view into a [CudaSlice] created by [CudaSlice::try_slice_mut()].
#[derive(Debug)]
pub struct GcuViewMut<'a, T> {
    pub(crate) root: &'a mut driv::topsDeviceptr_t,
    pub(crate) ptr: driv::topsDeviceptr_t,
    pub(crate) len: usize,
    marker: PhantomData<T>,
}

impl<T: DeviceCopy> GcuSlice<T> {
    /// Creates a [GcuViewMut] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `offset >= self.len`
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> GcuViewMut<'_, T> {
        self.try_slice_mut(range).unwrap()
    }

    /// Fallible version of [CudaSlice::slice_mut]
    pub fn try_slice_mut(&mut self, range: impl RangeBounds<usize>) -> Option<GcuViewMut<'_, T>> {
        range.bounds(..self.len).map(|(start, end)| GcuViewMut {
            // ptr: self.device_ptr + (start * std::mem::size_of::<T>()) as u64, 
            ptr: (self.device_ptr() as u64 + (start * std::mem::size_of::<T>()) as u64) as *mut c_void,
            // ptr: ((self.device_ptr().clone() as u64) + (start * std::mem::size_of::<T>() as u64)) as *mut c_void, //.offset(isize::try_from().unwrap())},
            root: self.buffer.as_device_ptr_mut().as_mut(),
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Reinterprets the slice of memory into a different type. `len` is the number
    /// of elements of the new type `S` that are expected. If not enough bytes
    /// are allocated in `self` for the view, then this returns `None`.
    ///
    /// # Safety
    /// This is unsafe because not the memory for the view may not be a valid interpretation
    /// for the type `S`.
    pub unsafe fn transmute_mut<S>(&mut self, len: usize) -> Option<GcuViewMut<'_, S>> {
        (len * std::mem::size_of::<S>() <= self.num_bytes()).then_some(GcuViewMut {
            ptr: self.buffer.as_device_ptr().as_raw(),
            root: self.buffer.as_device_ptr_mut().as_mut(),
            len,
            marker: PhantomData,
        })
    }
}

impl<'a, T> GcuViewMut<'a, T> {
    /// Creates a [CudaView] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `range.start >= self.len`
    pub fn slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> GcuView<'a, T> {
        self.try_slice(range).unwrap()
    }

    /// Fallible version of [GcuViewMut::slice]
    pub fn try_slice<'b: 'a>(&'b self, range: impl RangeBounds<usize>) -> Option<GcuView<'a, T>> {
        range.bounds(..self.len).map(|(start, end)| GcuView {
            root: self.root,
            ptr: (self.ptr as u64 + (start * std::mem::size_of::<T>()) as u64) as *mut c_void,
            len: end - start,
            marker: PhantomData,
        })
    }

    /// Creates a [GcuViewMut] at the specified offset from the start of `self`.
    ///
    /// Returns `None` if `offset >= self.len`
    pub fn slice_mut<'b: 'a>(&'b mut self, range: impl RangeBounds<usize>) -> GcuViewMut<'a, T> {
        self.try_slice_mut(range).unwrap()
    }

    /// Fallible version of [GcuViewMut::slice_mut]
    pub fn try_slice_mut<'b: 'a>(
        &'b mut self,
        range: impl RangeBounds<usize>,
    ) -> Option<GcuViewMut<'a, T>> {
        range.bounds(..self.len).map(|(start, end)| GcuViewMut {
            ptr: (self.ptr as u64 + (start * std::mem::size_of::<T>()) as u64) as *mut c_void,
            root: self.root,
            len: end - start,
            marker: PhantomData,
        })
    }
}

pub trait RangeHelper: RangeBounds<usize> {
    fn inclusive_start(&self, valid_start: usize) -> usize;
    fn exclusive_end(&self, valid_end: usize) -> usize;
    fn bounds(&self, valid: impl RangeHelper) -> Option<(usize, usize)> {
        let vs = valid.inclusive_start(0);
        let ve = valid.exclusive_end(usize::MAX);
        let s = self.inclusive_start(vs);
        let e = self.exclusive_end(ve);

        let inside = s >= vs && e <= ve;
        let valid = s < e || (s == e && !matches!(self.end_bound(), Bound::Included(_)));

        (inside && valid).then_some((s, e))
    }
}

impl<R: RangeBounds<usize>> RangeHelper for R {
    fn inclusive_start(&self, valid_start: usize) -> usize {
        match self.start_bound() {
            Bound::Included(n) => *n,
            Bound::Excluded(n) => *n + 1,
            Bound::Unbounded => valid_start,
        }
    }
    fn exclusive_end(&self, valid_end: usize) -> usize {
        match self.end_bound() {
            Bound::Included(n) => *n + 1,
            Bound::Excluded(n) => *n,
            Bound::Unbounded => valid_end,
        }
    }
}


impl<T: DeviceCopy> DeviceSlice<T> for GcuSlice<T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DeviceSlice<T> for GcuView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DeviceSlice<T> for GcuViewMut<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<T: DeviceCopy> DevicePtr<T> for GcuSlice<T> {
    fn device_ptr(&self) -> driv::topsDeviceptr_t {
        self.buffer.as_device_ptr().as_raw() as *mut c_void
    }
}

impl<'a, T> DevicePtr<T> for GcuView<'a, T> {
    fn device_ptr(&self) -> driv::topsDeviceptr_t {
        self.ptr
    }
}

impl<'a, T> DevicePtr<T> for GcuViewMut<'a, T> {
    fn device_ptr(&self) -> driv::topsDeviceptr_t {
        self.ptr
    }
}


impl<T: DeviceCopy> DevicePtrMut<T> for GcuSlice<T> {
    fn device_ptr_mut(&mut self) -> driv::topsDeviceptr_t {
        self.buffer.as_device_ptr().as_raw() as *mut c_void
    }
}

impl<'a, T> DevicePtrMut<T> for GcuViewMut<'a, T> {
    fn device_ptr_mut(&mut self) -> driv::topsDeviceptr_t {
        self.ptr
    }
}

// unsafe impl<T: DeviceCopy> DeviceCopy for &mut GcuSlice<T> {
//     #[inline(always)]
//     fn as_kernel_param(&self) -> *mut std::ffi::c_void {
//         (&self.device_ptr()) as *const driv::topsDeviceptr_t as *mut std::ffi::c_void
//     }
// }

unsafe impl<T: DeviceCopy> DeviceCopy for &GcuSlice<T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.device_ptr()) as *const driv::topsDeviceptr_t as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceCopy> DeviceCopy for &GcuView<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const driv::topsDeviceptr_t as *mut std::ffi::c_void
    }
}

unsafe impl<'a, T: DeviceCopy> DeviceCopy for &GcuViewMut<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const driv::topsDeviceptr_t as *mut std::ffi::c_void
    }
}

