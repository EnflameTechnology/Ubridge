use std::mem::MaybeUninit;
use tops::driv_eccl::lib;
use tops::driv_eccl::sys;
use tops::driv_eccl::{EcclError, EcclStatus};
use tops_backend as tops;

pub unsafe fn comm_destroy(comm: sys::ecclComm_t) -> Result<EcclStatus, EcclError> {
    lib().ecclCommDestroy(comm).result()
}

pub unsafe fn comm_abort(comm: sys::ecclComm_t) -> Result<EcclStatus, EcclError> {
    lib().ecclCommAbort(comm).result()
}

pub fn get_eccl_version() -> Result<::core::ffi::c_int, EcclError> {
    let mut version: ::core::ffi::c_int = 0;
    unsafe {
        lib().ecclGetVersion(&mut version).result()?;
    }
    Ok(version)
}

pub fn get_uniqueid() -> Result<sys::ecclUniqueId, EcclError> {
    let mut uniqueid = MaybeUninit::uninit();
    Ok(unsafe {
        lib().ecclGetUniqueId(uniqueid.as_mut_ptr()).result()?;
        uniqueid.assume_init()
    })
}

pub unsafe fn comm_init_rank(
    comm: *mut sys::ecclComm_t,
    nranks: ::core::ffi::c_int,
    comm_id: sys::ecclUniqueId,
    rank: ::core::ffi::c_int,
) -> Result<EcclStatus, EcclError> {
    lib().ecclCommInitRank(comm, nranks, comm_id, rank).result()
}

pub unsafe fn comm_init_all(
    comm: *mut sys::ecclComm_t,
    ndev: ::core::ffi::c_int,
    devlist: *const ::core::ffi::c_int,
) -> Result<EcclStatus, EcclError> {
    lib().ecclCommInitAll(comm, ndev, devlist).result()
}

pub unsafe fn comm_count(comm: sys::ecclComm_t) -> Result<::core::ffi::c_int, EcclError> {
    let mut count = 0;
    lib().ecclCommCount(comm, &mut count).result()?;
    Ok(count)
}

pub unsafe fn comm_cu_device(comm: sys::ecclComm_t) -> Result<::core::ffi::c_int, EcclError> {
    let mut device = 0;
    lib().ecclCommDevice(comm, &mut device).result()?;
    Ok(device)
}

pub unsafe fn comm_user_rank(comm: sys::ecclComm_t) -> Result<::core::ffi::c_int, EcclError> {
    let mut rank = 0;
    lib().ecclCommUserRank(comm, &mut rank).result()?;
    Ok(rank)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ecclDataType_t,
    op: sys::ecclRedOp_t,
    root: ::core::ffi::c_int,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclReduce(
            sendbuff,
            recvbuff,
            count as u64,
            datatype,
            op,
            root,
            comm,
            stream,
        )
        .result()
}

pub unsafe fn broadcast(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ecclDataType_t,
    root: ::core::ffi::c_int,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclBroadcast(
            sendbuff,
            recvbuff,
            count as u64,
            datatype,
            root,
            comm,
            stream,
        )
        .result()
}

pub unsafe fn all_reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ecclDataType_t,
    op: sys::ecclRedOp_t,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclAllReduce(sendbuff, recvbuff, count as u64, datatype, op, comm, stream)
        .result()
}

pub unsafe fn reduce_scatter(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    recvcount: usize,
    datatype: sys::ecclDataType_t,
    op: sys::ecclRedOp_t,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclReduceScatter(
            sendbuff,
            recvbuff,
            recvcount as u64,
            datatype,
            op,
            comm,
            stream,
        )
        .result()
}

pub unsafe fn all_gather(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    sendcount: usize,
    datatype: sys::ecclDataType_t,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclAllGather(sendbuff, recvbuff, sendcount as u64, datatype, comm, stream)
        .result()
}

pub unsafe fn send(
    sendbuff: *const ::core::ffi::c_void,
    count: usize,
    datatype: sys::ecclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclSend(sendbuff, count as u64, datatype, peer, comm, stream)
        .result()
}

pub unsafe fn recv(
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ecclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ecclComm_t,
    stream: tops::driv::topsStream_t,
) -> Result<EcclStatus, EcclError> {
    lib()
        .ecclRecv(recvbuff, count as u64, datatype, peer, comm, stream)
        .result()
}

pub fn group_end() -> Result<EcclStatus, EcclError> {
    unsafe { lib().ecclGroupEnd().result() }
}

pub fn group_start() -> Result<EcclStatus, EcclError> {
    unsafe { lib().ecclGroupStart().result() }
}
