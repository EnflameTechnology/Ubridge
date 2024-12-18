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
* @file    gcu_launch.rs
* @brief
*
* @author  Guoqing Bao
* @date    2023-09-05 - 2024-01-10
* @version V0.1
* @par     Copyright (c) Enflame Tech Company.
* @par     History: compatible with runtime 3.0 kernel launch
* @par     Comments: a gcu kernel launch abstraction.
*/
use crate::gcu_device::GcuFunction;
pub use cust_core::_hidden::DeviceCopy;
use std::ffi::c_void;
use std::ptr;
#[cfg(feature = "tops_backend")]
pub use tops::driv;
#[cfg(feature = "tops_backend")]
use tops::stream::TopsStream as Stream;
#[cfg(feature = "tops_backend")]
use tops_backend as tops;
use uhal::error::DeviceResult;

use tops::error::ToResult;

#[derive(Clone, Copy, Debug)]
pub struct GcuLaunchConfig {
    /// (width, height, depth) of grid in blocks
    pub grid_dim: (u32, u32, u32),

    /// (x, y, z) dimension of each thread block
    pub block_dim: (u32, u32, u32),

    /// Dynamic shared-memory size per thread block in bytes
    pub shared_mem_bytes: u32,
}

impl GcuLaunchConfig {
    pub fn set_shared_memory(&mut self, size_in_bytes: u32) {
        const SHARED_MEM_SIZE : u32 = 48 * 1024 * 1024;
        let shared_mem_bytes = ((size_in_bytes + 4095) / 4096) * 4096 + 4096;
        self.shared_mem_bytes = if shared_mem_bytes > SHARED_MEM_SIZE { SHARED_MEM_SIZE } else { shared_mem_bytes };
    }
    
    pub fn for_threds(n: u32) -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (n, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn for_ucopy() -> Self {
        Self {
            grid_dim: (1, 1, 1),
            block_dim: (Self::max_sip_num(), 1, 1),
            shared_mem_bytes: 0,
        }
    }

    #[cfg(feature = "scorpio")]
    pub fn max_sip_num() -> u32 {
        12
    }

    #[cfg(not(feature = "scorpio"))]
    pub fn max_sip_num() -> u32 {
        6
    }
}

/// Consumes a [GcuFunction] to execute asychronously on the device with
/// params determined by generic parameter `Params`.
///
/// This is impl'd multiple times for different number and types of params. In
/// general, `Params` should impl [DeviceCopy].
///
/// ```ignore
/// # let dev = GcuDevice::new(0).unwrap();
/// let my_kernel: GcuFunction = dev.get_func("my_module", "my_kernel").unwrap();
/// let cfg: GcuLaunchConfig = GcuLaunchConfig {
///     grid_dim: (1, 1, 1),
///     block_dim: (1, 1, 1),
///     shared_mem_bytes: 0,
/// };
/// let params = (1i32, 2u64, 3usize);
/// unsafe { my_kernel.launch(cfg, params) }.unwrap();
/// ```
///
/// # Safety
///
/// This is not safe really ever, because there's no garuntee that `Params`
/// will work for any [GcuFunction] passed in. Great care should be taken
/// to ensure that [GcuFunction] works with `Params` and that the correct
/// parameters have `&mut` in front of them.
///
/// Additionally, kernels can mutate data that is marked as immutable,
/// such as `&GcuSlice<T>`.
///
/// See [GcuLaunchAsync::launch] for more details
pub unsafe trait GcuLaunchAsync<Params> {
    /// Launches the [GcuFunction] with the corresponding `Params`.
    ///
    /// # Safety
    ///
    /// This method is **very** unsafe.
    ///
    /// See cuda documentation notes on this as well:
    /// <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#functions>
    ///
    /// 1. `params` can be changed regardless of `&` or `&mut` usage.
    /// 2. `params` will be changed at some later point after the
    /// function returns because the kernel is executed async.
    /// 3. There are no guaruntees that the `params`
    /// are the correct number/types/order for `func`.
    /// 4. Specifying the wrong values for [LaunchConfig] can result
    /// in accessing/modifying values past memory limits.
    ///
    /// ## Asynchronous mutation
    ///
    /// Since this library queues kernels to be launched on a single
    /// stream, and really the only way to modify [crate::driver::CudaSlice] is through
    /// kernels, mutating the same [crate::driver::CudaSlice] with multiple kernels
    /// is safe. This is because each kernel is executed sequentially
    /// on the stream.
    ///
    /// **Modifying a value on the host that is in used by a
    /// kernel is undefined behavior.** But is hard to do
    /// accidentally.
    ///
    /// Also for this reason, do not pass in any values to kernels
    /// that can be modified on the host. This is the reason
    /// [DeviceRepr] is not implemented for rust primitive
    /// references.
    ///
    /// ## Use after free
    ///
    /// Since the drop implementation for [crate::driver::CudaSlice] also occurs
    /// on the device's single stream, any kernels launched before
    /// the drop will complete before the value is actually freed.
    ///
    /// **If you launch a kernel or drop a value on a different stream
    /// this may not hold**
    unsafe fn launch(self, cfg: &GcuLaunchConfig, params: Params) -> DeviceResult<()>;

    /// Launch the function on a stream concurrent to the device's default
    /// work stream.
    ///
    /// # Safety
    /// This method is even more unsafe than [LaunchAsync::launch], all the same rules apply,
    /// except now things are executing in parallel to each other.
    ///
    /// That means that if any of the kernels modify the same memory location, you'll get race
    /// conditions or potentially undefined behavior.
    unsafe fn launch_on_stream(
        self,
        stream: &Stream,
        cfg: &GcuLaunchConfig,
        params: Params,
    ) -> DeviceResult<()>;
}

macro_rules! impl_launch {
    ([$($Vars:tt),*], [$($Idx:tt),*]) => {
unsafe impl<$($Vars: DeviceCopy),*> GcuLaunchAsync<($($Vars, )*)> for GcuFunction {
    #[inline(always)]
    unsafe fn launch(
        self,
        cfg: &GcuLaunchConfig,
        args: ($($Vars, )*)
    ) -> DeviceResult<()> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.launch_async_impl(&cfg, params)
    }

    #[inline(always)]
    unsafe fn launch_on_stream(
        self,
        stream: &Stream,
        cfg: &GcuLaunchConfig,
        args: ($($Vars, )*)
    ) -> DeviceResult<()> {
        let params = &mut [$(args.$Idx.as_kernel_param(), )*];
        self.par_launch_async_impl(stream, &cfg, params)
    }
}
    };
}

impl_launch!([A], [0]);
impl_launch!([A, B], [0, 1]);
impl_launch!([A, B, C], [0, 1, 2]);
impl_launch!([A, B, C, D], [0, 1, 2, 3]);
impl_launch!([A, B, C, D, E], [0, 1, 2, 3, 4]);
impl_launch!([A, B, C, D, E, F], [0, 1, 2, 3, 4, 5]);
impl_launch!([A, B, C, D, E, F, G], [0, 1, 2, 3, 4, 5, 6]);
impl_launch!([A, B, C, D, E, F, G, H], [0, 1, 2, 3, 4, 5, 6, 7]);
impl_launch!([A, B, C, D, E, F, G, H, I], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);
impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
);

impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
);

impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
);

impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
);

impl_launch!(
    [A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
);


impl GcuFunction {
    #[inline(always)]
    unsafe fn launch_async_impl(
        self,
        cfg: &GcuLaunchConfig,
        params: &mut [*mut std::ffi::c_void],
    ) -> DeviceResult<()> {
        if let Some(func) = self.func {
            // println!("Launch {} func {:?}!", self.name, func);
            // let mut args_ = Vec::new();
            // for i in 0..args.len(){
            //     let vaddress = std::mem::transmute::<*mut c_void, *mut *mut c_void>((*params)[i]);
            //     unsafe {args_.push(*vaddress);}

            // }

            // let mut size :usize = (std::mem::size_of::<c_ulonglong>() * (params.len() - 1) + std::mem::size_of::<usize>()) as usize;
            // let _config = vec![0x1 as *const c_void, params.as_mut_ptr() as *const _ as *mut c_void, 0x2 as *const c_void, &mut size as *const _ as *mut c_void, 0x3 as *const c_void];
            let null = ptr::null_mut();
            let stream = if self.is_async {
                self.stream.unwrap()
            } else {
                null
            };
            let ret = driv::topsModuleLaunchKernel(
                func,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream,
                params.as_mut_ptr(),
                null as *mut *mut c_void,
                // config.as_mut_ptr() as *mut *mut c_void
            )
            .to_result();

            if ret.is_err() {
                println!("Launch Error for function {}", self.func_name);
            }

            return ret;
        }
        Ok(())
    }

    #[inline(always)]
    unsafe fn par_launch_async_impl(
        self,
        _stream: &Stream,
        _cfg: &GcuLaunchConfig,
        _params: &mut [*mut std::ffi::c_void],
    ) -> DeviceResult<()> {
        // self.device.bind_to_thread()?;
        // launch_kernel(
        //     self.cu_function,
        //     cfg.grid_dim,
        //     cfg.block_dim,
        //     cfg.shared_mem_bytes,
        //     stream.stream,
        //     params,
        // )
        Ok(())
    }
}
