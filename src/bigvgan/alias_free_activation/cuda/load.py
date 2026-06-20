# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

import os
import pathlib
import subprocess
from typing import List, Tuple

import torch
from torch.utils import cpp_extension


def load():
    is_rocm = getattr(torch.version, "hip", None) is not None

    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    _create_build_dir(buildpath)

    sources = [
        str(srcpath / "anti_alias_activation.cpp"),
        str(srcpath / "anti_alias_activation_cuda.cu"),
    ]

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
        ] + _cuda_arch_flags()

    return cpp_extension.load(
        name="anti_alias_activation_cuda",
        sources=sources,
        build_directory=str(buildpath),
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=True,
    )


def _cuda_arch_flags() -> List[str]:
    if cpp_extension.CUDA_HOME is None:
        return []

    try:
        _, major_s, minor_s = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
        major = int(major_s)
        minor = int(minor_s)
    except Exception:
        return []

    archs: List[Tuple[int, int]] = [(7, 0), (7, 5)]

    if major >= 11:
        archs.append((8, 0))

    if major > 11 or (major == 11 and minor >= 1):
        archs.append((8, 6))

    if major > 11 or (major == 11 and minor >= 8):
        archs.extend([(8, 9), (9, 0)])

    flags: List[str] = []
    for major_cc, minor_cc in archs:
        cc = f"{major_cc}{minor_cc}"
        flags.extend(["-gencode", f"arch=compute_{cc},code=sm_{cc}"])

    if archs:
        major_cc, minor_cc = archs[-1]
        cc = f"{major_cc}{minor_cc}"
        flags.extend(["-gencode", f"arch=compute_{cc},code=compute_{cc}"])

    return flags


def _get_cuda_bare_metal_version(cuda_dir):
    if not cuda_dir:
        return "", "", ""

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


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            raise RuntimeError(f"Creation of the build directory {buildpath} failed")
