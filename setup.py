import os
import subprocess
from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
except ImportError:
    torch = None
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None


def get_cuda_bare_metal_version(cuda_dir):
    if not cuda_dir:
        return "", "", ""
    try:
        raw_output = subprocess.check_output(
            [os.path.join(cuda_dir, "bin", "nvcc"), "-V"],
            universal_newlines=True,
        )
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
        return raw_output, bare_metal_major, bare_metal_minor
    except Exception:
        return "", "", ""


def get_cuda_arch_flags():
    if CUDA_HOME is None:
        return []

    _, major_s, minor_s = get_cuda_bare_metal_version(CUDA_HOME)
    try:
        major = int(major_s)
        minor = int(minor_s)
    except ValueError:
        return []

    archs = [(7, 0), (7, 5)]

    if major >= 11:
        archs.append((8, 0))

    if major > 11 or (major == 11 and minor >= 1):
        archs.append((8, 6))

    if major > 11 or (major == 11 and minor >= 8):
        archs.extend([(8, 9), (9, 0)])

    flags = []
    for major_cc, minor_cc in archs:
        cc = f"{major_cc}{minor_cc}"
        flags.extend(["-gencode", f"arch=compute_{cc},code=sm_{cc}"])

    if archs:
        major_cc, minor_cc = archs[-1]
        cc = f"{major_cc}{minor_cc}"
        flags.extend(["-gencode", f"arch=compute_{cc},code=compute_{cc}"])

    return flags


ext_modules = []
cmdclass = {}

build_cuda_ext = os.environ.get("BUILD_CUDA_EXT", "0") == "1"

if (
    build_cuda_ext
    and torch is not None
    and (torch.cuda.is_available() or CUDA_HOME is not None)
):
    is_rocm = getattr(torch.version, "hip", None) is not None

    extra_cflags = ["-O3"]
    if is_rocm:
        extra_cuda_cflags = ["-O3"]
    else:
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ] + get_cuda_arch_flags()

    source_dir = os.path.join("src", "bigvgan", "alias_free_activation", "cuda")
    sources = [
        os.path.join(source_dir, "anti_alias_activation.cpp"),
        os.path.join(source_dir, "anti_alias_activation_cuda.cu"),
    ]

    ext_modules.append(
        CUDAExtension(
            name="bigvgan.anti_alias_activation_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": extra_cflags,
                "nvcc": extra_cuda_cflags,
            },
        )
    )

    cmdclass["build_ext"] = BuildExtension.with_options(use_ninja=True)

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
