use std::io::Write;
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../kernels/*.cpp");
    // println!("cargo:rerun-if-changed=cleanup.sh");
    cuda::build_dorado_kernels();
    cuda::build_pavo_kernels();
}

mod cuda {

    pub fn build_dorado_kernels() -> () {
        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("unary");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("transpose");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("activation");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("batch_matmul");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("convolution");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("element");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("fused_batch_matmul");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-dorado.sh")
            .arg("transposed_matmul");
        command.spawn().expect("failed").wait_with_output();
    }

    pub fn build_pavo_kernels() -> () {
        let mut command = std::process::Command::new("bash");
        command.arg("../tools/topscc-compile-pavo.sh").arg("unary");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("transpose");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("activation");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("batch_matmul");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("convolution");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("element");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("fused_batch_matmul");
        command.spawn().expect("failed").wait_with_output();

        let mut command = std::process::Command::new("bash");
        command
            .arg("../tools/topscc-compile-pavo.sh")
            .arg("transposed_matmul");
        command.spawn().expect("failed").wait_with_output();
    }
}
