use ubridge::tests::{network_test, kernel_test, gemm_test};
use uhal::error::DeviceResult;

fn main() -> DeviceResult<()> {

    match gemm_test::gemm_test() {
        Ok(()) => {
            println!("\nLaunch gemm_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch gemm_test failed.");
            return Err(e);
        }
    }

    match kernel_test::kernel_test() {
        Ok(()) => {
            println!("\nLaunch kernel_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch kernel_test failed.");
            return Err(e);
        }
    }

    match network_test::network_test() {
        Ok(()) => {
            println!("\nLaunch network_test successfully.");
        }
        Err(e) => {
            println!("\nLaunch network_test failed.");
            return Err(e);
        }
    }


    println!("\n\nPASSED!\n\n");
    Ok(())
}
