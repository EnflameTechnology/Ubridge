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
 */
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
