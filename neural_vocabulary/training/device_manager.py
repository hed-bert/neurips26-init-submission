"""Device abstraction for CUDA, MPS, Habana Gaudi, and CPU backends."""

from __future__ import annotations

import contextlib
import logging
from typing import overload

import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """Abstracts device differences between CUDA, MPS, Habana Gaudi, and CPU.

    All training code should use this class instead of raw device calls.

    Auto-detection priority (when device_type="auto"):
        CUDA > MPS > CPU
    """

    def __init__(self, device_type: str = "auto") -> None:
        if device_type == "auto":
            device_type = self._detect_best_device()

        # Normalize indexed device strings ("cuda:0", "mps:0", ...) to
        # their family name. Indexing is preserved on the resolved
        # ``torch.device`` below so a caller passing "cuda:1" still
        # targets the right GPU. Without this normalization, indexed
        # strings fell through to the unknown-family branch and the
        # encoder silently ran on CPU ( R3 hallu incident).
        original_device_type = device_type
        family = device_type.split(":", 1)[0]

        self._device_type = family
        self._use_lazy_mode = False

        if family == "hpu":
            try:
                import habana_frameworks.torch.core  # noqa: F401

                self._device = torch.device(original_device_type)
                self._use_lazy_mode = True
                logger.info("Using Habana Gaudi (HPU) with lazy mode")
            except ImportError:
                logger.warning("habana_frameworks not available, falling back to CPU")
                self._device_type = "cpu"
                self._device = torch.device("cpu")
        elif family == "cuda":
            if torch.cuda.is_available():
                self._device = torch.device(original_device_type)
                logger.info("Using CUDA: %s", torch.cuda.get_device_name(self._device))
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self._device_type = "cpu"
                self._device = torch.device("cpu")
        elif family == "mps":
            if torch.backends.mps.is_available():
                self._device = torch.device(original_device_type)
                logger.info("Using Apple MPS (Metal Performance Shaders)")
            else:
                logger.warning("MPS not available, falling back to CPU")
                self._device_type = "cpu"
                self._device = torch.device("cpu")
        elif family == "cpu":
            self._device_type = "cpu"
            self._device = torch.device("cpu")
            logger.info("Using CPU")
        else:
            # Refuse unknown device-family strings instead of silently
            # falling back to CPU. Caller misconfigurations (e.g.
            # "cuda0", typos) are now loud failures.
            raise ValueError(
                f"Unsupported device_type={original_device_type!r}. Use "
                "'auto', 'cuda', 'cuda:N', 'mps', 'mps:N', 'hpu', or 'cpu'."
            )

    @staticmethod
    def _detect_best_device() -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def device_type(self) -> str:
        return self._device_type

    def mark_step(self) -> None:
        """Required after loss.backward() and optimizer.step() in Gaudi lazy mode.

        No-op on CUDA and CPU.
        """
        if self._use_lazy_mode:
            import habana_frameworks.torch.core

            habana_frameworks.torch.core.mark_step()

    @property
    def amp_dtype(self) -> torch.dtype | None:
        """Return the AMP dtype for this device, or None if AMP is not used."""
        if self._device_type == "cuda":
            return torch.bfloat16
        if self._device_type == "mps":
            return torch.float16
        return None

    def get_amp_context(self) -> contextlib.AbstractContextManager:
        """Return the appropriate mixed-precision context manager."""
        if self._device_type == "hpu":
            from habana_frameworks.torch.hpex import hmp

            return hmp.disable_casts()
        if self._device_type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if self._device_type == "mps":
            return torch.amp.autocast(device_type="mps", dtype=torch.float16)
        return contextlib.nullcontext()

    def get_distributed_backend(self) -> str:
        """Return the distributed training backend name.

        MPS does not support distributed training; falls back to gloo.
        """
        if self._device_type == "hpu":
            return "hccl"
        if self._device_type == "cuda":
            return "nccl"
        return "gloo"

    @overload
    def to_device(self, obj: torch.Tensor) -> torch.Tensor: ...

    @overload
    def to_device(self, obj: torch.nn.Module) -> torch.nn.Module: ...

    def to_device(
        self, obj: torch.Tensor | torch.nn.Module
    ) -> torch.Tensor | torch.nn.Module:
        """Move a tensor or module to the managed device."""
        return obj.to(self._device)
