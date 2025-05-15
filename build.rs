use anyhow::{Context, Result};
use reqwest::blocking::Client;
use std::path::PathBuf;
use std::str::FromStr;

const KERNELS: [&str; 21] = [
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
    "mask_host"
];

fn unzip(filename: PathBuf, path: PathBuf) -> Result<()> {
    let mut command_tar = std::process::Command::new("tar");
    command_tar.arg("-xf");
    command_tar.arg(&filename);
    command_tar.arg("-C");
    command_tar.arg(&path);

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

fn check_atomic_op(path: PathBuf) -> Result<()> {
    let lib_file = format!(
        "topsacore_{:}-{:}.tar.gz",
        std::env::var("ATOMIC_TAG")?,
        std::env::var("ATOMIC_VERSION")?
    );
    let url = format!(
        "{:}/{:}/{:}",
        std::env::var("ATOMIC_URL")?,
        std::env::var("ATOMIC_VERSION")?,
        lib_file
    );
    let filename = path.join("atomic/".to_string() + &lib_file);

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
                    check_atomic_op(absolute_kernel_dir.clone())?;
                }
                let mut command = std::process::Command::new(compiler);
                if is_host_kernel {
                    command
                    .arg(kernel_file)
                    .arg(format!("-arch=gcu{compute_cap}"))
                    .arg("-O3")
                    .arg("-c")
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

    Ok(())
}
