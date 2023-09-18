use std::io::Write;
macro_rules! eprintln {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel in ["unary", "dot", "transpose", "activation", "element", "convolution", "batch_matmul"] {
        println!("cargo:rerun-if-changed=../kernels/{kernel}.cpp");
    }
    // println!("cargo:rerun-if-changed=../kernels/");
    // println!("cargo:rerun-if-changed=cleanup.sh");
    gcu::build_kernels();
    // gcu::build_pavo_kernels();
}

mod gcu {
    pub fn build_kernels() -> () {
        
        for kernel in ["unary", "dot", "transpose", "activation", "element", "convolution", "batch_matmul"] {
            let in_file = "../kernels/".to_string() + kernel + ".cpp";
            let in_filename = std::path::Path::new(&in_file);

            let out_file_dorado = "../kernels/dorado/".to_string() + kernel + ".topsfb";
            let output_filename = std::path::Path::new(&out_file_dorado);
            let ignore = if output_filename.exists() {
                let out_modified = output_filename.metadata().unwrap().modified().unwrap();
                if in_filename.exists() {
                    let in_modified = in_filename.metadata().unwrap().modified().unwrap();
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    true
                }

            } else{
                false
            };

            if !ignore {
                let mut command = std::process::Command::new("bash");
                command
                    .arg("../tools/topscc-compile-dorado.sh")
                    .arg(kernel);
                let _ = command.spawn().expect("failed").wait_with_output();
            }


            let out_file_pavo = "../kernels/pavo/".to_string() + kernel + ".topsfb";
            let output_filename = std::path::Path::new(&out_file_pavo);
            let ignore = if output_filename.exists() {
                let out_modified = output_filename.metadata().unwrap().modified().unwrap();
                if in_filename.exists() {
                    let in_modified = in_filename.metadata().unwrap().modified().unwrap();
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    true
                }

            } else{
                false
            };

            if !ignore {
                let mut command = std::process::Command::new("bash");
                command
                    .arg("../tools/topscc-compile-pavo.sh")
                    .arg(kernel);
                let _ = command.spawn().expect("failed").wait_with_output();
            }

        }
    }
}
