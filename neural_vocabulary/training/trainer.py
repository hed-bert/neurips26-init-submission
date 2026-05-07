"""Training loop for HED-BERT.

Device-agnostic trainer with W&B/TensorBoard logging, gradient clipping,
mixed precision, loss scheduling, and checkpointing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from neural_vocabulary.configs import HEDBERTConfig, save_config

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from neural_vocabulary.data.hed_vectorizer import HEDVectorizer
    from neural_vocabulary.training.device_manager import DeviceManager

logger = logging.getLogger(__name__)


class HEDBERTTrainer:
    """Device-agnostic training loop for HED-BERT.

    Supports CUDA, MPS, Habana Gaudi, and CPU backends via DeviceManager.
    Optional W&B integration for experiment tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: HEDBERTConfig,
        device_manager: DeviceManager,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        wandb_run: Any | None = None,
        hed_vectorizer: HEDVectorizer | None = None,
        tb_writer: Any | None = None,
        max_steps_per_epoch: int | None = None,
    ) -> None:
        self.config = config
        self.dm = device_manager
        self.device_type = self.dm.device_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_run = wandb_run
        self.hed_vectorizer = hed_vectorizer
        self.tb_writer = tb_writer
        self.max_steps_per_epoch = max_steps_per_epoch

        # Move model and loss to device
        self.model = self.dm.to_device(model)
        self.loss_fn = self.dm.to_device(loss_fn)

        # Mixed precision via DeviceManager (CUDA, MPS, Gaudi all supported)
        # GradScaler is only useful for fp16; bf16 has sufficient dynamic range
        self.use_scaler = (
            self.device_type == "cuda" and self.dm.amp_dtype != torch.bfloat16
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_scaler else None

        # Optimizer: include both model and loss_fn parameters so that
        # learnable heads inside the loss (e.g. HED prediction MLP) are trained.
        all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        self.optimizer = AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler: OneCycleLR (warmup + cosine annealing in one)
        try:
            steps_per_ep = len(train_loader)
        except TypeError:
            # IterableDataset (streaming) has no len; use max_steps_per_epoch
            steps_per_ep = self.max_steps_per_epoch or 100
        total_steps = steps_per_ep * config.total_epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.warmup_ratio,
            anneal_strategy="cos",
        )

        self.global_step = 0

    def train(
        self,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 10,
    ) -> dict[str, float]:
        """Run full training for config.total_epochs epochs.

        Args:
            checkpoint_dir: Directory to save periodic checkpoints. If None,
                no checkpoints are saved during training.
            checkpoint_every: Save a checkpoint every N epochs.

        Returns:
            Final epoch's training metrics as a dict of loss names to values.
        """
        try:
            n_batches = len(self.train_loader)
        except TypeError:
            n_batches = self.max_steps_per_epoch or -1
        logger.info(
            "Starting training: %d epochs, %d steps/epoch, device=%s",
            self.config.total_epochs,
            n_batches,
            self.device_type,
        )

        metrics: dict[str, float] = {}
        for epoch in range(self.config.total_epochs):
            epoch_start = time.monotonic()
            metrics = self.train_epoch(epoch)
            elapsed = time.monotonic() - epoch_start

            log_msg = (
                f"Epoch {epoch + 1}/{self.config.total_epochs} "
                f"({elapsed:.1f}s) - "
                + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )

            val_metrics: dict[str, float] = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
                log_msg += " | val: " + ", ".join(
                    f"{k}: {v:.4f}" for k, v in val_metrics.items()
                )

            logger.info(log_msg)

            # TensorBoard logging
            if self.tb_writer is not None:
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(f"train/{k}", v, self.global_step)
                self.tb_writer.add_scalar(
                    "train/epoch_time_s", elapsed, self.global_step
                )
                self.tb_writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )
                for k, v in val_metrics.items():
                    self.tb_writer.add_scalar(f"val/{k}", v, self.global_step)
                # GPU memory stats
                if self.device_type == "cuda":
                    mem = torch.cuda.memory_allocated() / 1e9
                    mem_max = torch.cuda.max_memory_allocated() / 1e9
                    self.tb_writer.add_scalar("gpu/memory_gb", mem, self.global_step)
                    self.tb_writer.add_scalar(
                        "gpu/peak_memory_gb", mem_max, self.global_step
                    )
                    try:
                        util = torch.cuda.utilization()
                        self.tb_writer.add_scalar(
                            "gpu/utilization_pct", util, self.global_step
                        )
                    except ModuleNotFoundError:
                        pass  # nvidia-ml-py not installed

            # W&B logging
            if self.wandb_run is not None:
                epoch_log = {f"train/{k}": v for k, v in metrics.items()}
                epoch_log["epoch"] = epoch
                epoch_log["epoch_time_s"] = elapsed
                for k, v in val_metrics.items():
                    epoch_log[f"val/{k}"] = v
                self.wandb_run.log(epoch_log, step=self.global_step)

            # Notify dataset of epoch end (for cache rotation)
            if hasattr(self.train_loader.dataset, "notify_epoch_end"):
                self.train_loader.dataset.notify_epoch_end()  # type: ignore[call-non-callable]

            # Periodic checkpointing
            if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_dir, epoch)

        # Final checkpoint
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir, self.config.total_epochs - 1)

        return metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: current epoch number (0-indexed).

        Returns:
            Dict of average losses for this epoch.
        """
        # Update loss phase for curriculum scheduling
        if hasattr(self.loss_fn, "update_phase"):
            self.loss_fn.update_phase(epoch + 1)  # type: ignore[call-non-callable]

        self.model.train()
        self.loss_fn.train()

        running: dict[str, float] = {}
        n_steps = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            losses, grad_norm = self._train_step(batch)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.detach().item()
            n_steps += 1
            self.global_step += 1

            # For streaming/iterable datasets, limit steps per epoch
            if self.max_steps_per_epoch and n_steps >= self.max_steps_per_epoch:
                break

            if self.wandb_run is not None and n_steps % 10 == 0:
                # Log every 10 steps to reduce GPU-CPU sync overhead
                step_log = {f"step/{k}": v.detach().item() for k, v in losses.items()}
                step_log["step/learning_rate"] = self.optimizer.param_groups[0]["lr"]
                step_log["step/grad_norm"] = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                )
                self.wandb_run.log(step_log, step=self.global_step)

        if n_steps == 0:
            logger.warning("Epoch %d: no training steps executed", epoch)
            return {}

        return {k: v / n_steps for k, v in running.items()}

    def validate(self) -> dict[str, float]:
        """Run validation.

        Returns:
            Dict of average validation losses.

        Raises:
            ValueError: If no validation loader was provided.
        """
        if self.val_loader is None:
            raise ValueError("No validation loader provided")

        self.model.eval()
        self.loss_fn.eval()

        running: dict[str, float] = {}
        n_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_batch(batch)

                with self.dm.get_amp_context():
                    outputs = self.model(batch)
                    recon_target = outputs.get("reconstruction_target", batch["eeg"])
                    losses = self.loss_fn(
                        reconstructed=outputs["reconstructed"],
                        original=recon_target,
                        evt_embeddings=outputs.get("evt_embeddings"),
                        hed_targets=batch.get("hed_targets"),
                        recon_mask=None,
                    )

                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + v.item()
                n_steps += 1

        if n_steps == 0:
            logger.warning("Validation: no steps executed")
            return {}

        return {k: v / n_steps for k, v in running.items()}

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save model state_dict and config for reproducibility.

        Only saves model weights (no optimizer state) for cross-platform
        compatibility per project conventions.

        Args:
            path: directory to save checkpoint files.
            epoch: current epoch number.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = path / f"epoch_{epoch:04d}.pt"
        torch.save(self.model.state_dict(), checkpoint_file)

        # Save loss module (contains HED prediction head weights)
        loss_file = path / f"loss_epoch_{epoch:04d}.pt"
        torch.save(self.loss_fn.state_dict(), loss_file)

        config_file = path / "config.yaml"
        save_config(self.config, config_file)

        logger.info("Saved checkpoint: %s (epoch %d)", checkpoint_file, epoch)

    def load_checkpoint(self, path: Path) -> int:
        """Load a checkpoint and return the epoch number.

        Args:
            path: path to a .pt checkpoint file. The filename is expected
                to follow the pattern epoch_NNNN.pt.

        Returns:
            The epoch number extracted from the filename.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            ValueError: If the epoch number cannot be parsed from the filename.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state_dict = torch.load(path, map_location=self.dm.device, weights_only=True)
        self.model.load_state_dict(state_dict)

        # Parse epoch from filename: epoch_0042.pt -> 42
        stem = path.stem
        try:
            epoch = int(stem.split("_")[1])
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Cannot parse epoch from checkpoint filename: {path.name}"
            ) from e

        logger.info("Loaded checkpoint: %s (epoch %d)", path, epoch)
        return epoch

    def _train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | float]:
        """Execute a single training step.

        Args:
            batch: dict with at least 'eeg' key, optionally 'hed_targets'.

        Returns:
            Tuple of (losses dict, gradient norm).
        """
        self.optimizer.zero_grad()

        # Use pre-computed HED vectors if available, else vectorize strings
        hed_targets = batch.pop("hed_targets", None)
        if hed_targets is not None:
            hed_targets = hed_targets.to(batch["eeg"].device)
        else:
            hed_tag_strings = batch.pop("hed_tags", None)
            if hed_tag_strings is not None and self.hed_vectorizer is not None:
                vectors = []
                for tag in hed_tag_strings:
                    if tag is not None:
                        vectors.append(self.hed_vectorizer.vectorize(tag))
                    else:
                        vectors.append(torch.zeros(self.hed_vectorizer.vocab_size))
                hed_targets = torch.stack(vectors).to(batch["eeg"].device)

        with self.dm.get_amp_context():
            # Use packed forward if epoch_boundaries present
            if "token_types" in batch:
                outputs = self.model.forward_packed(batch)  # type: ignore[call-non-callable]
                evt_emb = outputs["evt_embeddings"]  # (B, max_n_evts, embed)
                evt_epoch_indices = outputs.get("evt_epoch_indices")  # (B, max_evts)
                n_valid_evts = outputs.get("n_valid_evts")  # (B,)

                # Vectorized HED target alignment (no Python loops)
                hed_targets_packed = batch.get("hed_targets_packed")
                if (
                    hed_targets_packed is not None
                    and evt_epoch_indices is not None
                    and n_valid_evts is not None
                ):
                    hed_targets_packed = hed_targets_packed.to(evt_emb.device)
                    n_evts = n_valid_evts  # (B,)
                    n_batch = min(evt_emb.shape[0], hed_targets_packed.shape[0])
                    max_evts = evt_emb.shape[1]
                    max_epochs = hed_targets_packed.shape[1]

                    # Clamp indices to valid range
                    idx = evt_epoch_indices[:n_batch, :max_evts].clamp(
                        0, max_epochs - 1
                    )

                    # Gather HED targets for each EVT position: (B, max_evts, vocab)
                    idx_expanded = idx.unsqueeze(-1).expand(
                        -1, -1, hed_targets_packed.shape[2]
                    )
                    gathered_hed = torch.gather(
                        hed_targets_packed[:n_batch], 1, idx_expanded
                    )

                    # Validity mask: EVT exists AND HED is non-zero
                    evt_mask = torch.arange(max_evts, device=evt_emb.device).unsqueeze(
                        0
                    ) < n_evts[:n_batch].unsqueeze(1)
                    hed_nonzero = gathered_hed.sum(dim=-1) > 0
                    valid = evt_mask & hed_nonzero  # (B, max_evts)

                    if valid.any():
                        evt_valid = evt_emb[:n_batch][valid]
                        hed_valid = gathered_hed[valid]
                    else:
                        evt_valid = evt_emb[:, 0]
                        hed_valid = None
                else:
                    if hed_targets_packed is None:
                        logger.debug("No hed_targets_packed in batch")
                    evt_valid = evt_emb[:, 0]
                    hed_valid = None
            else:
                outputs = self.model(batch)
                evt_valid = outputs.get("evt_embeddings")
                hed_valid = hed_targets

            # Task codes for task_codes prediction target
            task_codes_valid = None
            task_codes_packed = batch.get("task_codes_packed")
            if (
                task_codes_packed is not None
                and "token_types" in batch
                and evt_epoch_indices is not None
                and n_valid_evts is not None
            ):
                task_codes_packed = task_codes_packed.to(evt_emb.device)
                n_batch = min(evt_emb.shape[0], task_codes_packed.shape[0])
                max_evts = evt_emb.shape[1]
                max_tc_epochs = task_codes_packed.shape[1]

                tc_idx = evt_epoch_indices[:n_batch, :max_evts].clamp(
                    0, max_tc_epochs - 1
                )
                gathered_tc = torch.gather(
                    task_codes_packed[:n_batch], 1, tc_idx
                )  # (B, max_evts)

                # Valid: EVT exists AND task_code >= 0
                evt_mask_tc = torch.arange(max_evts, device=evt_emb.device).unsqueeze(
                    0
                ) < n_valid_evts[:n_batch].unsqueeze(1)
                tc_valid_mask = evt_mask_tc & (gathered_tc >= 0)

                if tc_valid_mask.any():
                    evt_valid = evt_emb[:n_batch][tc_valid_mask]
                    task_codes_valid = gathered_tc[tc_valid_mask]
                    # Override hed_valid to None for task_codes mode
                    hed_valid = None

            recon_target = outputs.get("reconstruction_target", batch["eeg"])
            losses = self.loss_fn(
                reconstructed=outputs["reconstructed"],
                original=recon_target,
                evt_embeddings=evt_valid,
                hed_targets=hed_valid,
                recon_mask=outputs.get("recon_mask"),
                task_codes=task_codes_valid,
            )

        total_loss = losses["total"]

        if self.use_scaler and self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        self.dm.mark_step()

        all_params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params, self.config.max_grad_norm
        )

        if self.use_scaler and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.dm.mark_step()
        self.scheduler.step()

        return losses, grad_norm

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensors in a batch dict to the managed device."""
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.dm.device, non_blocking=True)
            else:
                moved[k] = v
        return moved
