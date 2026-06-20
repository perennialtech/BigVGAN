# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

from typing import Any

import torch
import torch.nn as nn

from ..torch.resample import UpSample1d, DownSample1d
from . import load

try:
    anti_alias_activation_cuda = load.load()
except Exception as exc:
    raise RuntimeError(
        "Failed to load the fused alias-free activation CUDA extension. "
        "The CUDA alias-free activation path is explicit and does not fall back "
        "to the unfused PyTorch implementation. Install a compatible CUDA/NVCC/"
        "PyTorch toolchain or disable use_cuda_kernel."
    ) from exc


_EPS = 1.0e-9


class FusedAliasFreeActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        up_filter: torch.Tensor,
        down_filter: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        alpha_logscale: bool,
    ) -> torch.Tensor:
        if not inputs.is_cuda:
            raise ValueError("Fused alias-free activation requires CUDA input.")
        if inputs.dim() != 3:
            raise ValueError(
                f"Fused alias-free activation expects [B, C, T], got {tuple(inputs.shape)}."
            )
        if not inputs.is_contiguous():
            raise ValueError("Fused alias-free activation requires contiguous input.")
        if inputs.size(2) <= 0:
            raise ValueError("Fused alias-free activation requires T > 0.")
        if inputs.device.index != torch.cuda.current_device():
            raise ValueError(
                f"Input tensor is on CUDA device {inputs.device.index}, but current "
                f"CUDA device is {torch.cuda.current_device()}."
            )
        if alpha.dim() != 1 or alpha.numel() != inputs.size(1):
            raise ValueError(
                f"alpha must be 1D with C={inputs.size(1)} elements, got {tuple(alpha.shape)}."
            )
        if beta.dim() != 1 or beta.numel() != inputs.size(1):
            raise ValueError(
                f"beta must be 1D with C={inputs.size(1)} elements, got {tuple(beta.shape)}."
            )

        up_filter_f = (
            up_filter.reshape(-1)
            .to(device=inputs.device, dtype=torch.float32)
            .contiguous()
        )
        down_filter_f = (
            down_filter.reshape(-1)
            .to(device=inputs.device, dtype=torch.float32)
            .contiguous()
        )

        if up_filter_f.numel() != 12 or down_filter_f.numel() != 12:
            raise ValueError(
                "Fused alias-free activation supports only 12-tap up/down filters."
            )

        alpha_f = alpha.to(device=inputs.device, dtype=torch.float32).contiguous()
        beta_f = beta.to(device=inputs.device, dtype=torch.float32).contiguous()

        if alpha_logscale:
            alpha_eff = torch.exp(alpha_f).contiguous()
            beta_eff = torch.exp(beta_f).contiguous()
        else:
            alpha_eff = alpha_f
            beta_eff = beta_f

        inv_beta = torch.reciprocal(beta_eff + _EPS).contiguous()
        alpha_over_beta = (alpha_eff * inv_beta).contiguous()

        outputs = anti_alias_activation_cuda.forward(
            inputs,
            up_filter_f,
            down_filter_f,
            alpha_eff,
            inv_beta,
        )

        ctx.save_for_backward(
            inputs,
            up_filter_f,
            down_filter_f,
            alpha_eff,
            beta_eff,
            inv_beta,
            alpha_over_beta,
        )
        ctx.alpha_logscale = alpha_logscale
        ctx.input_dtype = inputs.dtype
        ctx.alpha_dtype = alpha.dtype
        ctx.beta_dtype = beta.dtype

        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor):
        (
            inputs,
            up_filter,
            down_filter,
            alpha_eff,
            beta_eff,
            inv_beta,
            alpha_over_beta,
        ) = ctx.saved_tensors

        grad_outputs = grad_outputs.contiguous()

        grad_input_f32, grad_alpha_eff, grad_inv_beta = (
            anti_alias_activation_cuda.backward(
                grad_outputs,
                inputs,
                up_filter,
                down_filter,
                alpha_eff,
                inv_beta,
                alpha_over_beta,
            )
        )

        grad_beta_eff = -grad_inv_beta * inv_beta * inv_beta

        if ctx.alpha_logscale:
            grad_alpha = grad_alpha_eff * alpha_eff
            grad_beta = grad_beta_eff * beta_eff
        else:
            grad_alpha = grad_alpha_eff
            grad_beta = grad_beta_eff

        grad_input = grad_input_f32.to(dtype=ctx.input_dtype)
        grad_alpha = grad_alpha.to(dtype=ctx.alpha_dtype)
        grad_beta = grad_beta.to(dtype=ctx.beta_dtype)

        return grad_input, None, None, grad_alpha, grad_beta, None


class AliasFreeActivationCuda(nn.Module):
    """
    Fully fused CUDA alias-free activation.

    This is not a silent optional acceleration wrapper. If this class is used,
    the CUDA extension must be available and the configuration must exactly match
    the supported fused contract:

    - input shape: contiguous [B, C, T]
    - up_ratio = 2
    - down_ratio = 2
    - up_kernel_size = 12
    - down_kernel_size = 12
    - CUDA input tensor on the current CUDA device
    - input dtype: float32, float16, or bfloat16
    - internal FIR/activation accumulation: float32

    Forward and backward are implemented by the custom CUDA extension.
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()

        if up_ratio != 2:
            raise ValueError(
                f"Fused CUDA alias-free activation supports up_ratio=2, got {up_ratio}."
            )
        if down_ratio != 2:
            raise ValueError(
                f"Fused CUDA alias-free activation supports down_ratio=2, got {down_ratio}."
            )
        if up_kernel_size != 12:
            raise ValueError(
                f"Fused CUDA alias-free activation supports up_kernel_size=12, got {up_kernel_size}."
            )
        if down_kernel_size != 12:
            raise ValueError(
                f"Fused CUDA alias-free activation supports down_kernel_size=12, got {down_kernel_size}."
            )
        if not hasattr(activation, "alpha"):
            raise TypeError("activation must expose an alpha parameter.")
        if not hasattr(activation, "alpha_logscale"):
            raise TypeError("activation must expose alpha_logscale.")

        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = getattr(self.act, "beta", self.act.alpha)

        return FusedAliasFreeActivationFunction.apply(
            x,
            self.upsample.filter,
            self.downsample.lowpass.filter,
            self.act.alpha,
            beta,
            bool(self.act.alpha_logscale),
        )


Activation1d = AliasFreeActivationCuda
