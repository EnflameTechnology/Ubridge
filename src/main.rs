use ubridge::tests::{network_test, memory_test, kernel_test};
use uhal::error::DeviceResult;

fn main() -> DeviceResult<()> {
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
