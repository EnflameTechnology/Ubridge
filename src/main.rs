use ubridge::tests::{network_test, kernel_test, gemm_test};
use uhal::error::DeviceResult;
use ubridge::gemm_tuner::{AtenGemmTuner, AtenGemmInfo, AtenGemmTune, TopsopDataType};

fn main() -> DeviceResult<()> {
    let mut info = AtenGemmInfo::default();
    info.M = 16;
    info.batch = 16;
    info.is_batch = true;
    let mut tune = AtenGemmTune::default();
    let tuner = AtenGemmTuner::new();
    tuner.tuner(&info, &mut tune);

    gemm_test::gemm_test();
    println!("{:?}", tune);

    match kernel_test::kernel_test() {
        Ok(()) => {
            println!("\nLaunched kernel_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch kernel_test failed.");
            return Err(e);
        }
    }

    match network_test::network_test() {
        Ok(()) => {
            println!("\nLaunched network_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch network_test failed.");
            return Err(e);
        }
    }
    println!("\n\nPASSED!\n\n");
    Ok(())
}
