import os

from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ImportError:
    torch = None
    BuildExtension = None
    CUDAExtension = None
    CUDA_HOME = None


def get_cuda_arch_flags():
    return ["-gencode", "arch=compute_75,code=sm_75"]


ext_modules = []
cmdclass = {}

build_cuda_ext = os.environ.get("BUILD_CUDA_EXT", "0") == "1"

if build_cuda_ext:
    if torch is None or BuildExtension is None or CUDAExtension is None:
        raise RuntimeError(
            "BUILD_CUDA_EXT=1 was set, but torch and torch.utils.cpp_extension "
            "could not be imported."
        )

    if CUDA_HOME is None and not torch.cuda.is_available():
        raise RuntimeError(
            "BUILD_CUDA_EXT=1 was set, but no CUDA toolkit was found. "
            "Set CUDA_HOME or CUDA_PATH to the CUDA Toolkit install directory."
        )

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
