use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::path::PathBuf;
use std::str::FromStr;
const BC_FILE_NAME: &str = "acore.bc";

const KERNELS: [&str; 30] = [
    "unary",
    "fill",
    "binary",
    "affine",
    "cast",
    "reduce",
    "ternary",
    "indexing",
    "matmul",
    "embedding",
    "kvconcat",
    "conv",
    "copy",
    "quant",
    "cache",
    "attention",
    "sort",
    "topk_host",
    "moe_host",
    "dequant_host",
    "mask_host",
    "moe_align_host",
    "fused_moe_host",
    "topk_softmax_host",
    "causal_conv1d_host",
    "gdn_gating_host",
    "gdn_l2norm_host",
    "gdn_rmsnorm_host",
    "gdn_recurrence_host",
    "gdn_scatter_host",
];

fn unzip(filename: PathBuf, path: PathBuf) -> Result<()> {
    let mut command_tar = std::process::Command::new("tar");
    command_tar.arg("-xf");
    command_tar.arg(&filename);
    command_tar.arg("-C");
    command_tar.arg(&path);

    std::fs::create_dir_all(path)?;
    let output = command_tar
        .spawn()
        .context(format!("failed spawning tar"))?
        .wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!(
            "tar error while executing: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
            &command_tar,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }
    Ok(())
}

fn check_atomic_op(path: PathBuf, kernel_dir: PathBuf) -> Result<()> {
    let url = format!("{:}/{}", std::env::var("ATOMIC_URL")?, BC_FILE_NAME,);
    let filename = if kernel_dir.join(BC_FILE_NAME).exists() {
        kernel_dir.join(BC_FILE_NAME)
    } else {
        path.join("atomic/".to_string() + BC_FILE_NAME)
    };

    if !filename.exists() {
        let _ = std::fs::create_dir(path.join("atomic/"));
        let client = Client::new();
        match client.get(&url).send() {
            Ok(mut response) => {
                let mut file = std::fs::File::create(&filename)?;
                std::io::copy(&mut response, &mut file)?;
                unzip(filename, path.join("atomic/"))?
            }
            _ => {
                anyhow::bail!(
                    "error while configuring atomic dependencies, unable to obtain {:?}",
                    url,
                )
            }
        }
    } else if !path.join("atomic/include").exists() || !path.join("atomic/lib").exists() {
        unzip(filename, path.join("atomic/"))?
    }
    Ok(())
}

fn main() -> Result<()> {
    let platform = "scorpio";
    let kernel_dir = PathBuf::from("kernels/");
    let absolute_kernel_dir = std::fs::canonicalize(&kernel_dir).unwrap();
    let kernel_out_dir = absolute_kernel_dir.join(platform.to_string() + "/");
    let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
        |_| num_cpus::get_physical(),
        |s| usize::from_str(&s).unwrap(),
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus)
        .build_global()
        .unwrap();

    println!("cargo:rerun-if-changed=build.rs");

    for kernel_file in KERNELS.iter() {
        println!("cargo:rerun-if-changed=kernels/{kernel_file}.cpp");
    }

    let mut build_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);
    if !std::env::set_current_dir(&build_dir).is_ok() {
        build_dir = PathBuf::from(absolute_kernel_dir.clone());
    }

    let compute_cap = 300 as usize;
    let compiler = "/opt/tops/bin/topscc";
    let mut first_atomic_check = true;
    KERNELS
        .iter()
        .map(|f| {
            let fname = f.to_string();
            let is_host_kernel = fname.find("_host").is_some();
            let fatbin_file = if is_host_kernel {
                build_dir.join(format!("{}.a", f.to_string()))
            } else {
                kernel_out_dir.join(f.to_string() + ".topsfb")
            };
            let kernel_file = absolute_kernel_dir.join(f.to_string() + ".cpp");
            let out_modified: Result<_, _> = fatbin_file.metadata().and_then(|m| m.modified());
            let should_compile = if fatbin_file.exists() {
                let in_modified = kernel_file.metadata().unwrap().modified().unwrap();
                in_modified.duration_since(out_modified.unwrap()).is_ok()
            } else {
                true
            };
            let build_file = if is_host_kernel {
                build_dir.join(f.to_string() + ".o")
            } else {
                build_dir.join(f.to_string() + ".topsfb")
            };
            if should_compile {
                if first_atomic_check {
                    first_atomic_check = false;
                    check_atomic_op(absolute_kernel_dir.clone(), kernel_out_dir.clone())?;
                }
                let mut command = std::process::Command::new(compiler);
                if is_host_kernel {
                    command
                    .arg(kernel_file)
                    .arg(format!("-arch=gcu{compute_cap}"))
                    .arg("-O3")
                    .arg("-c")
                    .arg("-std=c++17")
                    .arg("-fPIC")
                    .arg("-ltops")
                    .arg("-Xclang")
                    .arg("-fallow-half-arguments-and-returns")
                    .args(["-o", build_file.to_str().unwrap()])
                    .arg(format!("-D__GCU_ARCH__={compute_cap}"))
                    .arg(format!("-D__KRT_ARCH__={compute_cap}"))
                    .arg("-D__ATOMIC_OP")
                    .arg("-D__ACORE_OP__")
                    .arg("-fno-omit-frame-pointer")
                    .arg("-DNDEBUG")
                    .arg(format!("-I{:}", absolute_kernel_dir.join("atomic/include").to_str().unwrap()))
                    .arg(format!("-I{:}", absolute_kernel_dir.join("atomic/include/common").to_str().unwrap()))
                    .arg(format!("--tops-device-lib-path={:}", absolute_kernel_dir.join("atomic/lib").to_str().unwrap()))
                    .arg("--tops-device-lib=libacoreop.bc");

                    let output = command
                        .spawn()
                        .context(format!("failed spawning {compiler}"))?
                        .wait_with_output()?;
                    if !output.status.success() {
                        anyhow::bail!(
                            "{:?} error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                            compiler,
                            &command,
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        )
                    }
                    let mut command_link = std::process::Command::new("ar");
                    command_link
                        .arg("-crv")
                        .args(["-o", fatbin_file.to_str().unwrap()])
                        .arg(build_file);
                    let output_linker = command_link
                        .spawn()
                        .context("failed spawning linker")?
                        .wait_with_output()?;
                    if !output_linker.status.success() {
                        anyhow::bail!(
                            "{:?} error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                            compiler,
                            &command,
                            String::from_utf8_lossy(&output_linker.stdout),
                            String::from_utf8_lossy(&output_linker.stderr)
                        )
                    }
                } else {
                    command
                    .arg(kernel_file)
                    .arg(format!("-arch=gcu{compute_cap}"))
                    .arg("-O3")
                    .arg("-std=c++17")
                    .arg("-fPIC")
                    .arg("-ltops")
                    .arg("-Xclang")
                    .arg("-fallow-half-arguments-and-returns")
                    .args(["-o", build_file.to_str().unwrap()])
                    .arg(format!("-D__GCU_ARCH__={compute_cap}"))
                    .arg(format!("-D__KRT_ARCH__={compute_cap}"))
                    .arg("-D__ATOMIC_OP")
                    .arg("-D__ACORE_OP__")
                    .arg("-DTOPS_DISABLE_FORCE_INLINE")
                    .arg("--target=x86_64-unknown-linux-gnu")
                    .arg("-fno-omit-frame-pointer")
                    .arg("-DNDEBUG")
                    .arg(format!("-I{:}", absolute_kernel_dir.join("atomic/include").to_str().unwrap()))
                    .arg(format!("-I{:}", absolute_kernel_dir.join("atomic/include/common").to_str().unwrap()))
                    .arg(format!("--tops-device-lib-path={:}", absolute_kernel_dir.join("atomic/lib").to_str().unwrap()))
                    .arg("--tops-device-lib=libacoreop.bc")
                    .arg("--save-temps");

                    let output = command
                        .spawn()
                        .context(format!("failed spawning {compiler}"))?
                        .wait_with_output()?;
                    let tmp_fatbin_file = build_dir.join(f.to_string() + ".cpp-tops-dtu-enflame-tops.topsfb");
                    if !output.status.success() && !PathBuf::from(tmp_fatbin_file).exists() {
                        anyhow::bail!(
                            "{:?} error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                            compiler,
                            &command,
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        )
                    }
                    let mut command_mv = std::process::Command::new("mv");
                    command_mv.arg(build_dir.join(f.to_string() + ".cpp-tops-dtu-enflame-tops.topsfb"));
                    command_mv.arg(fatbin_file.clone());
                    let output = command_mv
                        .spawn()
                        .context(format!("failed spawning {compiler}"))?
                        .wait_with_output()?;
                    if !output.status.success() {
                        anyhow::bail!(
                            "{:?} error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                            compiler,
                            &command_mv,
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        )
                    }
                }
            }
            if is_host_kernel {
                let mut command_cp = std::process::Command::new("cp");
                command_cp.arg(build_dir.join(fatbin_file.clone()));
                command_cp.arg(build_dir.join(format!("lib{}.a", fname)));
                let output = command_cp
                    .spawn()
                    .context(format!("failed spawning {compiler}"))?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "{:?} error while linking host kernel: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        compiler,
                        &fname,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                } else {
                    println!("cargo:rustc-link-search={}", build_dir.display());
                    println!("cargo:rustc-link-lib={}", fname);
                }
            }
            Ok(())
        })
        .collect::<Result<()>>()?;

    // Build topsaten C++ wrappers (regular C++ compiled with g++, linked
    // against libtopsaten.so).
    // MoE wrapper is always built; ops wrapper only with "aten" feature.
    {
        let topsaten_include =
            std::env::var("TOPSATEN_HOME").unwrap_or_else(|_| "/usr".to_string());
        let topsaten_include_path = format!("{}/include/gcu", topsaten_include);

        let mut wrappers: Vec<(&str, &str)> =
            vec![("topsaten_moe_wrapper", "libtopsaten_moe_wrapper")];
        #[cfg(feature = "aten")]
        wrappers.push(("topsaten_ops_wrapper", "libtopsaten_ops_wrapper"));

        for (src_name, lib_name) in wrappers {
            let wrapper_src = absolute_kernel_dir.join(format!("{}.cpp", src_name));
            let wrapper_obj = build_dir.join(format!("{}.o", src_name));
            let wrapper_lib = build_dir.join(format!("{}.a", lib_name));
            println!("cargo:rerun-if-changed=kernels/{}.cpp", src_name);

            let should_compile = if wrapper_lib.exists() {
                let in_mod = wrapper_src.metadata().unwrap().modified().unwrap();
                let out_mod = wrapper_lib.metadata().unwrap().modified().unwrap();
                in_mod.duration_since(out_mod).is_ok()
            } else {
                true
            };

            if should_compile {
                let mut cmd = std::process::Command::new("g++");
                cmd.arg("-c")
                    .arg("-std=c++17")
                    .arg("-O2")
                    .arg("-fPIC")
                    .arg("-DNDEBUG")
                    .arg(format!("-I{}", topsaten_include_path))
                    .arg("-I/opt/tops/include")
                    .arg("-o")
                    .arg(wrapper_obj.to_str().unwrap())
                    .arg(wrapper_src.to_str().unwrap());

                let output = cmd
                    .spawn()
                    .context(format!("failed spawning g++ for {}", src_name))?
                    .wait_with_output()?;
                if !output.status.success() {
                    anyhow::bail!(
                        "g++ error compiling {}:\n# stdout\n{}\n# stderr\n{}",
                        src_name,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    );
                }

                let mut ar_cmd = std::process::Command::new("ar");
                ar_cmd
                    .arg("-crv")
                    .arg(wrapper_lib.to_str().unwrap())
                    .arg(wrapper_obj.to_str().unwrap());

                let ar_output = ar_cmd
                    .spawn()
                    .context(format!("failed spawning ar for {}", src_name))?
                    .wait_with_output()?;
                if !ar_output.status.success() {
                    anyhow::bail!(
                        "ar error creating {}.a:\n# stderr\n{}",
                        lib_name,
                        String::from_utf8_lossy(&ar_output.stderr)
                    );
                }
            }

            println!("cargo:rustc-link-search={}", build_dir.display());
            println!("cargo:rustc-link-lib=static={}", &lib_name[3..]); // strip "lib" prefix
        }

        let topsaten_lib_dir =
            std::env::var("TOPSATEN_HOME").unwrap_or_else(|_| "/usr".to_string());
        println!("cargo:rustc-link-search=native={}/lib", topsaten_lib_dir);
        println!("cargo:rustc-link-lib=dylib=topsaten");
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    Ok(())
}
