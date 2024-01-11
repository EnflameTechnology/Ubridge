use std::io::Write;
macro_rules! eprintln {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for kernel in ["unary", "fill", "binary", "affine", "cast", 
            "reduce", "ternary", "indexing", "matmul", "embedding", "kvconcat", "conv"] {
        println!("cargo:rerun-if-changed=../kernels/{kernel}.cpp");
    }
    gcu::build_kernels();
}

mod gcu {
    pub fn build_kernels() -> () {
        #[cfg(feature = "tops_backend")]
        let platform = "pavo"; //default kernel path

        #[cfg(feature = "dorado")]
        let platform = "dorado"; 

        #[cfg(feature = "scorpio")]
        let platform = "scorpio"; 

        // for platform in ["pavo", "dorado", "scorpio"] {
            for kernel in ["unary", "matmul", "fill", "binary", "affine", 
                    "cast", "reduce", "ternary", "indexing", "embedding", "kvconcat", "conv"] {
                let in_file = "../kernels/".to_string() + kernel + ".cpp";
                let in_filename = std::path::Path::new(&in_file);

                let out_file = "../kernels/".to_string() + platform + "/" + kernel + ".topsfb";
                let output_filename = std::path::Path::new(&out_file);
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
                        .arg("../tools/topscc-compile-".to_string() + platform + ".sh")
                        .arg(kernel);
                    let _ = command.spawn().expect("failed").wait_with_output();
                }
            }
        // }
        
    }
}
