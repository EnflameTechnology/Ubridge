use crate::device_ptr::{DevicePtr, DevicePtrMut};
use crate::eccllib;
#[allow(unused_imports)]
use crate::eccllib::{group_end, group_start};
use crate::gcu_device::GcuDevice;
use std::mem::MaybeUninit;
use std::ptr;
use std::{sync::Arc, vec, vec::Vec};
use tops::driv_eccl::sys;
use tops::driv_eccl::{EcclError, EcclStatus};
use tops_backend as tops;
use half;

#[derive(Debug, Clone)]
pub struct Comm {
    comm: sys::ecclComm_t,
    device: Arc<GcuDevice>,
    rank: usize,
    world_size: usize,
}

unsafe impl Send for Comm {}

#[derive(Debug, Clone, Copy)]
pub struct Id {
    id: sys::ecclUniqueId,
}

impl Id {
    pub fn new() -> Result<Self, EcclError> {
        let id = eccllib::get_uniqueid()?;
        Ok(Self { id })
    }

    pub fn uninit(internal: [::core::ffi::c_char; 128usize]) -> Self {
        let id = sys::ecclUniqueId { internal };
        Self { id }
    }

    pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
        &self.id.internal
    }
}

pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

fn convert_to_eccl_reduce_op(op: &ReduceOp) -> sys::ecclRedOp_t {
    match op {
        ReduceOp::Sum => sys::ecclRedOp_t::ecclSum,
        ReduceOp::Prod => sys::ecclRedOp_t::ecclProd,
        ReduceOp::Max => sys::ecclRedOp_t::ecclMax,
        ReduceOp::Min => sys::ecclRedOp_t::ecclMin,
        ReduceOp::Avg => sys::ecclRedOp_t::ecclAvg,
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        // TODO(thenerdstation): Shoule we instead do finalize then destory?
        unsafe {
            eccllib::comm_abort(self.comm).expect("Error when aborting Comm.");
        }
    }
}

pub trait EcclType {
    fn as_eccl_type() -> sys::ecclDataType_t;
}

macro_rules! define_eccl_type {
    ($t:ty, $eccl_type:expr) => {
        impl EcclType for $t {
            fn as_eccl_type() -> sys::ecclDataType_t {
                $eccl_type
            }
        }
    };
}

define_eccl_type!(f32, sys::ecclDataType_t::ecclFloat32);
define_eccl_type!(f64, sys::ecclDataType_t::ecclFloat64);
define_eccl_type!(i8, sys::ecclDataType_t::ecclInt8);
define_eccl_type!(i32, sys::ecclDataType_t::ecclInt32);
define_eccl_type!(i64, sys::ecclDataType_t::ecclInt64);
define_eccl_type!(u8, sys::ecclDataType_t::ecclUint8);
define_eccl_type!(u32, sys::ecclDataType_t::ecclUint32);
define_eccl_type!(u64, sys::ecclDataType_t::ecclUint64);
define_eccl_type!(char, sys::ecclDataType_t::ecclUint8);
define_eccl_type!(half::f16, sys::ecclDataType_t::ecclFloat16);
define_eccl_type!(half::bf16, sys::ecclDataType_t::ecclBfloat16);
impl Comm {
    /// Primitive to create new communication link on a single thread.
    /// WARNING: You are likely to get limited throughput using a single core
    /// to control multiple GPUs
    /// ```
    /// # use cudarc::driver::safe::{GcuDevice};
    /// # use cudarc::eccl::safe::{Comm, ReduceOp, group_start, group_end};
    /// let n = 2;
    /// let n_devices = GcuDevice::count().unwrap() as usize;
    /// let devices : Vec<_> = (0..n_devices).flat_map(GcuDevice::new).collect();
    /// let comms = Comm::from_devices(devices).unwrap();
    /// group_start().unwrap();
    /// (0..n_devices).map(|i| {
    ///     let comm = &comms[i];
    ///     let dev = comm.device();
    ///     let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
    ///     let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    ///     comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///         .unwrap();

    /// });
    /// group_start().unwrap();
    /// ```
    pub fn from_devices(devices: Vec<Arc<GcuDevice>>) -> Result<Vec<Self>, EcclError> {
        let n_devices = devices.len();
        let mut comms = vec![std::ptr::null_mut(); n_devices];
        let ordinals: Vec<_> = devices.iter().map(|d| d.id as i32).collect();
        unsafe {
            eccllib::comm_init_all(comms.as_mut_ptr(), n_devices as i32, ordinals.as_ptr())?;
        }

        let comms: Vec<Self> = comms
            .into_iter()
            .zip(devices.iter().cloned())
            .enumerate()
            .map(|(rank, (comm, device))| Self {
                comm,
                device,
                rank,
                world_size: n_devices,
            })
            .collect();

        Ok(comms)
    }

    pub fn device(&self) -> Arc<GcuDevice> {
        self.device.clone()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Primitive to create new communication link on each process (threads are possible but not
    /// recommended).
    ///
    /// WARNING: If using threads, uou are likely to get limited throughput using a single core
    /// to control multiple GPUs. Cuda drivers effectively use a global mutex thrashing
    /// performance on multi threaded multi GPU.
    /// ```
    /// let n = 2;
    /// let n_devices = 1; // This is to simplify this example.
    /// // Spawn this only on rank 0
    /// let id = Id::new().unwrap();
    /// // Send id.internal() to other ranks
    /// // let id = Id::uninit(id.internal().clone()); on other ranks
    ///
    /// let rank = 0;
    /// let dev = GcuDevice::new(rank).unwrap();
    /// let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();
    /// let slice = dev.htod_copy(vec![(rank + 1) as f32 * 1.0; n]).unwrap();
    /// let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    /// comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///     .unwrap();

    /// let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

    /// assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
    /// ```
    pub fn from_rank(
        device: Arc<GcuDevice>,
        rank: usize,
        world_size: usize,
        id: Id,
    ) -> Result<Self, EcclError> {
        let mut comm = MaybeUninit::uninit();

        let comm = unsafe {
            eccllib::comm_init_rank(
                comm.as_mut_ptr(),
                world_size
                    .try_into()
                    .expect("World_size cannot be casted to i32"),
                id.id,
                rank.try_into().expect("Rank cannot be cast to i32"),
            )?;
            comm.assume_init()
        };
        Ok(Self {
            comm,
            device,
            rank,
            world_size,
        })
    }
}

impl Comm {
    pub fn send<S: DevicePtr<T>, T: EcclType>(&self, data: &S, peer: i32) -> Result<(), EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::send(
                data.device_ptr() as *mut _,
                data.len(),
                T::as_eccl_type(),
                peer,
                self.comm,
                stream as *mut _,
            )?;
        }
        Ok(())
    }

    pub fn recv<R: DevicePtrMut<T>, T: EcclType>(
        &self,
        buff: &mut R,
        peer: i32,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::recv(
                buff.device_ptr_mut() as *mut _,
                buff.len(),
                T::as_eccl_type(),
                peer,
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn broadcast<S: DevicePtr<T>, R: DevicePtrMut<T>, T: EcclType>(
        &self,
        sendbuff: &Option<S>,
        recvbuff: &mut R,
        root: i32,
    ) -> Result<EcclStatus, EcclError> {
        unsafe {
            let send_ptr = match sendbuff {
                Some(buffer) => buffer.device_ptr() as *mut _,
                None => ptr::null(),
            };
            let stream = match self.device.stream_inner() {
                Some(s) => s,
                _ => ptr::null_mut(),
            };
            eccllib::broadcast(
                send_ptr,
                recvbuff.device_ptr_mut() as *mut _,
                recvbuff.len(),
                T::as_eccl_type(),
                root,
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn broadcast_in_place<R: DevicePtrMut<T>, T: EcclType>(
        &self,
        recvbuff: &mut R,
        root: i32,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::broadcast(
                recvbuff.device_ptr_mut() as *const _,
                recvbuff.device_ptr_mut() as *mut _,
                recvbuff.len(),
                T::as_eccl_type(),
                root,
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn all_gather<S: DevicePtr<T>, R: DevicePtrMut<T>, T: EcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::all_gather(
                sendbuff.device_ptr() as *mut _,
                recvbuff.device_ptr_mut() as *mut _,
                sendbuff.len(),
                T::as_eccl_type(),
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn all_reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: EcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::all_reduce(
                sendbuff.device_ptr() as *mut _,
                recvbuff.device_ptr_mut() as *mut _,
                sendbuff.len(),
                T::as_eccl_type(),
                convert_to_eccl_reduce_op(reduce_op),
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: EcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
        root: i32,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::reduce(
                sendbuff.device_ptr() as *mut _,
                recvbuff.device_ptr_mut() as *mut _,
                sendbuff.len(),
                T::as_eccl_type(),
                convert_to_eccl_reduce_op(reduce_op),
                root,
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn reduce_in_place<R: DevicePtrMut<T>, T: EcclType>(
        &self,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
        root: i32,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::reduce(
                recvbuff.device_ptr_mut() as *mut _,
                recvbuff.device_ptr_mut() as *mut _,
                recvbuff.len(),
                T::as_eccl_type(),
                convert_to_eccl_reduce_op(reduce_op),
                root,
                self.comm,
                stream as *mut _,
            )
        }
    }

    pub fn reduce_scatter<S: DevicePtr<T>, R: DevicePtrMut<T>, T: EcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
    ) -> Result<EcclStatus, EcclError> {
        let stream = match self.device.stream_inner() {
            Some(s) => s,
            _ => ptr::null_mut(),
        };
        unsafe {
            eccllib::reduce_scatter(
                sendbuff.device_ptr() as *mut _,
                recvbuff.device_ptr_mut() as *mut _,
                recvbuff.len(),
                T::as_eccl_type(),
                convert_to_eccl_reduce_op(reduce_op),
                self.comm,
                stream as *mut _,
            )
        }
    }
}

#[macro_export]
macro_rules! group {
    ($x:block) => {
        unsafe {
            eccllib::group_start().unwrap();
        }
        $x
        unsafe {
            eccllib::group_end().unwrap();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "no-std")]
    use no_std_compat::println;

    #[test]
    fn test_all_reduce() {
        let n = 2;
        let n_devices = GcuDevice::count().unwrap() as usize;
        let id = Id::new().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                println!("III {i}");
                std::thread::spawn(move || {
                    println!("Within thread {i}");
                    let dev = GcuDevice::new(i).unwrap();
                    let comm = Comm::from_rank(dev.clone(), i, n_devices, id).unwrap();
                    let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
                    comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
                        .unwrap();

                    let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

                    assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()
        }
    }

    #[test]
    fn test_all_reduce_views() {
        let n = 2;
        let n_devices = GcuDevice::count().unwrap() as usize;
        let id = Id::new().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                println!("III {i}");
                std::thread::spawn(move || {
                    println!("Within thread {i}");
                    let dev = GcuDevice::new(i).unwrap();
                    let comm = Comm::from_rank(dev.clone(), i, n_devices, id).unwrap();
                    let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
                    let slice_view = slice.slice(..);
                    let mut slice_receive_view = slice_receive.slice_mut(..);

                    comm.all_reduce(&slice_view, &mut slice_receive_view, &ReduceOp::Sum)
                        .unwrap();

                    let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

                    assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()
        }
    }
}
