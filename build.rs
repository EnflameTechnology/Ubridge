use anyhow::{Context, Result};
use rayon::prelude::*;
use reqwest::blocking::Client;
use std::path::{Path, PathBuf};
use std::str::FromStr;
const BC_FILE_NAME: &str = "acore.bc";

const KERNELS: [&str; 35] = [
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
    "gdn_ffi_bridge_host",
    "gdn_gating_host",
    "gdn_l2norm_host",
    "gdn_rmsnorm_host",
    "gdn_recurrence_host",
    "gdn_recurrence_varlen_host",
    "gdn_decode_slots_host",
    "gdn_decode_slots_gqa_host",
    "gdn_decode_recurrence_fused_bf16_host",
    "causal_conv1d_host",
    "cache_host",
];

/// Minimum expected size of extracted libacoreop.bc (~305MB). Used to detect
/// truncated extracts from a previous interrupted unpack.
const LIBACOREOP_MIN_BYTES: u64 = 200 * 1024 * 1024;

fn atomic_lib_ready(path: &Path) -> bool {
    let bc = path.join("atomic/lib/libacoreop.bc");
    match bc.metadata() {
        Ok(m) => m.len() >= LIBACOREOP_MIN_BYTES,
        Err(_) => false,
    }
}

fn unzip(filename: &Path, path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    // Block until tar fully finishes — do not start topscc until this returns.
    let output = std::process::Command::new("tar")
        .arg("-xf")
        .arg(filename)
        .arg("-C")
        .arg(path)
        .output()
        .context("failed spawning tar")?;
    if !output.status.success() {
        anyhow::bail!(
            "tar error while extracting {:?}:\n# stdout\n{}\n# stderr\n{}",
            filename,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    // Ensure data is durable before topscc mmaps libacoreop.bc.
    let _ = std::process::Command::new("sync").status();
    Ok(())
}

fn check_atomic_op(path: &Path, kernel_dir: &Path) -> Result<()> {
    let atomic_dir = path.join("atomic");
    let bc_path = atomic_dir.join("lib/libacoreop.bc");

    if atomic_lib_ready(path) && atomic_dir.join("include").exists() {
        println!(
            "cargo:warning=atomic lib ready: {} ({} bytes)",
            bc_path.display(),
            bc_path.metadata()?.len()
        );
        return Ok(());
    }

    // Incomplete/missing extract from a prior failed build — wipe and redo.
    if atomic_dir.exists() && !atomic_lib_ready(path) {
        println!(
            "cargo:warning=incomplete atomic extract at {}, re-extracting",
            atomic_dir.display()
        );
        let _ = std::fs::remove_dir_all(&atomic_dir);
    }

    let tarball = if kernel_dir.join(BC_FILE_NAME).exists() {
        kernel_dir.join(BC_FILE_NAME)
    } else {
        atomic_dir.join(BC_FILE_NAME)
    };

    if !tarball.exists() {
        std::fs::create_dir_all(&atomic_dir)?;
        let url = format!(
            "{}/{}",
            std::env::var("ATOMIC_URL")
                .context("ATOMIC_URL not set and kernels/scorpio/acore.bc missing")?,
            BC_FILE_NAME
        );
        let client = Client::new();
        let mut response = client
            .get(&url)
            .send()
            .with_context(|| format!("failed downloading {url}"))?;
        let mut file = std::fs::File::create(&tarball)?;
        std::io::copy(&mut response, &mut file)?;
        file.sync_all()?;
        drop(file);
    }

    println!(
        "cargo:warning=extracting atomic deps from {} → {} (blocking until done)",
        tarball.display(),
        atomic_dir.display()
    );
    unzip(&tarball, &atomic_dir)?;

    if !atomic_lib_ready(path) {
        anyhow::bail!(
            "atomic extract finished but {} missing or too small (need >= {} bytes)",
            bc_path.display(),
            LIBACOREOP_MIN_BYTES
        );
    }
    println!(
        "cargo:warning=atomic extract complete: {} ({} bytes)",
        bc_path.display(),
        bc_path.metadata()?.len()
    );
    Ok(())
}

/// Compile one kernel. Returns `Some(link_name)` for host kernels that need
/// `cargo:rustc-link-lib`, or `None` for device fatbins.
fn compile_kernel(
    name: &str,
    absolute_kernel_dir: &Path,
    kernel_out_dir: &Path,
    build_dir: &Path,
    compiler: &str,
    compute_cap: usize,
) -> Result<Option<String>> {
    let is_host_kernel = name.contains("_host");
    let fatbin_file = if is_host_kernel {
        build_dir.join(format!("{name}.a"))
    } else {
        kernel_out_dir.join(format!("{name}.topsfb"))
    };
    let kernel_file = absolute_kernel_dir.join(format!("{name}.cpp"));
    let should_compile = if fatbin_file.exists() {
        let in_modified = kernel_file.metadata()?.modified()?;
        let out_modified = fatbin_file.metadata()?.modified()?;
        in_modified.duration_since(out_modified).is_ok()
    } else {
        true
    };
    let build_file = if is_host_kernel {
        build_dir.join(format!("{name}.o"))
    } else {
        // Unique per-kernel temp dir avoids --save-temps collisions under parallel topscc.
        let tmp_dir = build_dir.join(format!("tmp_{name}"));
        std::fs::create_dir_all(&tmp_dir)?;
        tmp_dir.join(format!("{name}.topsfb"))
    };

    if should_compile {
        if is_host_kernel {
            let output = std::process::Command::new(compiler)
                .arg(&kernel_file)
                .arg(format!("-arch=gcu{compute_cap}"))
                .arg("-O3")
                .arg("-c")
                .arg("-std=c++17")
                .arg("-fPIC")
                .arg("-ltops")
                .arg("-Xclang")
                .arg("-fallow-half-arguments-and-returns")
                .arg("-o")
                .arg(&build_file)
                .arg(format!("-D__GCU_ARCH__={compute_cap}"))
                .arg(format!("-D__KRT_ARCH__={compute_cap}"))
                .arg("-D__ATOMIC_OP")
                .arg("-D__ACORE_OP__")
                .arg("-DTOPSCC_PRIVATE_DTE_AUTO_INIT")
                .arg("-fno-omit-frame-pointer")
                .arg("-DNDEBUG")
                .arg(format!("-I{}", absolute_kernel_dir.display()))
                .arg(format!(
                    "-I{}",
                    absolute_kernel_dir.join("atomic/include").display()
                ))
                .arg(format!(
                    "-I{}",
                    absolute_kernel_dir.join("atomic/include/common").display()
                ))
                .arg(format!(
                    "--tops-device-lib-path={}",
                    absolute_kernel_dir.join("atomic/lib").display()
                ))
                .arg("--tops-device-lib=libacoreop.bc")
                .output()
                .with_context(|| format!("failed spawning {compiler} for host kernel `{name}`"))?;
            if !output.status.success() {
                anyhow::bail!(
                    "host kernel `{name}` compile failed (status {:?}):\n# stdout\n{}\n# stderr\n{}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }

            let output_linker = std::process::Command::new("ar")
                .arg("-crv")
                .arg("-o")
                .arg(&fatbin_file)
                .arg(&build_file)
                .output()
                .with_context(|| format!("failed spawning ar for host kernel `{name}`"))?;
            if !output_linker.status.success() {
                anyhow::bail!(
                    "host kernel `{name}` ar failed (status {:?}):\n# stdout\n{}\n# stderr\n{}",
                    output_linker.status.code(),
                    String::from_utf8_lossy(&output_linker.stdout),
                    String::from_utf8_lossy(&output_linker.stderr)
                );
            }
        } else {
            let tmp_dir = build_file.parent().unwrap();
            let output = std::process::Command::new(compiler)
                .current_dir(tmp_dir)
                .arg(&kernel_file)
                .arg(format!("-arch=gcu{compute_cap}"))
                .arg("-O3")
                .arg("-std=c++17")
                .arg("-fPIC")
                .arg("-ltops")
                .arg("-Xclang")
                .arg("-fallow-half-arguments-and-returns")
                .arg("-o")
                .arg(&build_file)
                .arg(format!("-D__GCU_ARCH__={compute_cap}"))
                .arg(format!("-D__KRT_ARCH__={compute_cap}"))
                .arg("-D__ATOMIC_OP")
                .arg("-D__ACORE_OP__")
                .arg("-DTOPS_DISABLE_FORCE_INLINE")
                .arg("--target=x86_64-unknown-linux-gnu")
                .arg("-fno-omit-frame-pointer")
                .arg("-DNDEBUG")
                .arg(format!(
                    "-I{}",
                    absolute_kernel_dir.join("atomic/include").display()
                ))
                .arg(format!(
                    "-I{}",
                    absolute_kernel_dir.join("atomic/include/common").display()
                ))
                .arg(format!(
                    "--tops-device-lib-path={}",
                    absolute_kernel_dir.join("atomic/lib").display()
                ))
                .arg("--tops-device-lib=libacoreop.bc")
                .arg("--save-temps")
                .output()
                .with_context(|| format!("failed spawning {compiler} for kernel `{name}`"))?;

            let tmp_fatbin = tmp_dir.join(format!("{name}.cpp-tops-dtu-enflame-tops.topsfb"));
            if !output.status.success() && !tmp_fatbin.exists() {
                anyhow::bail!(
                    "device kernel `{name}` compile failed (status {:?}):\n# stdout\n{}\n# stderr\n{}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let src_fatbin = if tmp_fatbin.exists() {
                tmp_fatbin
            } else {
                build_file.clone()
            };
            std::fs::rename(&src_fatbin, &fatbin_file).with_context(|| {
                format!(
                    "failed moving {} → {} for kernel `{name}`",
                    src_fatbin.display(),
                    fatbin_file.display()
                )
            })?;
        }
    }

    if is_host_kernel {
        let lib_dst = build_dir.join(format!("lib{name}.a"));
        std::fs::copy(&fatbin_file, &lib_dst).with_context(|| {
            format!(
                "failed copying {} → {} for host kernel `{name}`",
                fatbin_file.display(),
                lib_dst.display()
            )
        })?;
        return Ok(Some(name.to_string()));
    }
    Ok(None)
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
    if std::env::set_current_dir(&build_dir).is_err() {
        build_dir = PathBuf::from(absolute_kernel_dir.clone());
    }

    let compute_cap = 300usize;
    let compiler = "/opt/tops/bin/topscc";

    // Unpack atomic bitcode first and wait until it is fully on disk.
    // Each new ubridge git rev uses a fresh cargo checkout, so this must
    // complete before any topscc invocation that loads libacoreop.bc.
    check_atomic_op(&absolute_kernel_dir, &kernel_out_dir)?;

    println!(
        "cargo:warning=compiling {} kernels in parallel ({} threads)",
        KERNELS.len(),
        num_cpus
    );

    // Parallel compile; collect every failure so cargo shows the real stderr.
    let results: Vec<(&str, Result<Option<String>>)> = KERNELS
        .par_iter()
        .map(|name| {
            let res = compile_kernel(
                name,
                &absolute_kernel_dir,
                &kernel_out_dir,
                &build_dir,
                compiler,
                compute_cap,
            );
            (*name, res)
        })
        .collect();

    let mut errors: Vec<String> = Vec::new();
    let mut link_libs: Vec<String> = Vec::new();
    for (name, res) in results {
        match res {
            Ok(Some(lib)) => link_libs.push(lib),
            Ok(None) => {}
            Err(e) => {
                // {:#} keeps the full anyhow chain + topscc stdout/stderr.
                errors.push(format!("=== kernel `{name}` ===\n{e:#}"));
            }
        }
    }

    if !errors.is_empty() {
        anyhow::bail!(
            "{} kernel(s) failed to compile:\n\n{}",
            errors.len(),
            errors.join("\n\n")
        );
    }

    // Emit link lines serially (cargo build-script protocol is not thread-safe).
    if !link_libs.is_empty() {
        println!("cargo:rustc-link-search={}", build_dir.display());
        for lib in link_libs {
            println!("cargo:rustc-link-lib={lib}");
        }
    }

    // Build topsaten C++ wrappers (regular C++ compiled with g++, linked
    // against libtopsaten.so).
    // MoE wrapper is always built; ops wrapper only with "aten" feature.
    #[cfg(feature = "aten")]
    {
        let topsaten_include =
            std::env::var("TOPSATEN_HOME").unwrap_or_else(|_| "/usr".to_string());
        let topsaten_include_path = format!("{}/include/gcu", topsaten_include);

        let mut wrappers: Vec<(&str, &str)> =
            vec![("topsaten_moe_wrapper", "libtopsaten_moe_wrapper")];
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
