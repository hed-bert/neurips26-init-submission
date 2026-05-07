"""Baseline feature extractors for comparison against HEDBERT [EVT] embeddings.

Three baselines that produce fixed-size feature vectors from EEG epochs:
1. RandomEmbeddingBaseline: random vectors with same dimensionality
2. PCABaseline: PCA of raw EEG epoch (top-k components as features)
3. MeanPowerSpectrumBaseline: mean power spectrum across channels

These baselines feed into the same linear probing pipeline as HEDBERT
embeddings, establishing a performance floor.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class RandomEmbeddingBaseline:
    """Generate random feature vectors with matched dimensionality.

    Provides a chance-level baseline for linear probing.
    """

    def __init__(self, embed_dim: int = 128, seed: int = 42) -> None:
        self.embed_dim = embed_dim
        self.seed = seed

    def extract(self, eeg_epochs: list[np.ndarray]) -> np.ndarray:
        """Generate random embeddings for each epoch.

        Args:
            eeg_epochs: List of (channels, time) arrays.

        Returns:
            (N, embed_dim) random feature matrix.
        """
        rng = np.random.RandomState(self.seed)
        return rng.randn(len(eeg_epochs), self.embed_dim).astype(np.float32)


class PCABaseline:
    """Top-k PCA components of flattened EEG epoch.

    Flattens each epoch to a 1D vector, fits PCA on training data,
    and transforms both train and eval epochs.
    """

    def __init__(self, n_components: int = 128) -> None:
        self.n_components = n_components
        self._pca: PCA | None = None
        self._target_len: int = 0

    def _flatten_epochs(
        self, eeg_epochs: list[np.ndarray], target_len: int | None = None
    ) -> np.ndarray:
        """Flatten variable-length epochs to fixed-size vectors.

        Truncates or zero-pads each (channels, time) epoch to a common
        length, then flattens to 1D.

        Args:
            eeg_epochs: List of (channels, time) arrays.
            target_len: Fixed time length. If None, uses median of input.
        """
        if not eeg_epochs:
            return np.empty((0, 0), dtype=np.float32)

        if target_len is None:
            lengths = [e.shape[1] for e in eeg_epochs]
            target_len = int(np.median(lengths))
        n_channels = eeg_epochs[0].shape[0]

        flat = np.zeros((len(eeg_epochs), n_channels * target_len), dtype=np.float32)
        for i, epoch in enumerate(eeg_epochs):
            t = min(epoch.shape[1], target_len)
            flat[i, : n_channels * t] = epoch[:, :t].ravel()

        return flat

    def fit(self, eeg_epochs: list[np.ndarray]) -> None:
        """Fit PCA on training epochs.

        Args:
            eeg_epochs: List of (channels, time) arrays.
        """
        flat = self._flatten_epochs(eeg_epochs)
        if flat.shape[0] == 0 or flat.shape[1] == 0:
            raise ValueError("Cannot fit PCA: empty input")

        # Store target_len so transform uses the same feature dimension
        lengths = [e.shape[1] for e in eeg_epochs]
        self._target_len = int(np.median(lengths))

        n_components = min(self.n_components, flat.shape[0], flat.shape[1])
        self._pca = PCA(n_components=n_components)
        self._pca.fit(flat)
        logger.info(
            "PCA fitted: %d components, %.1f%% variance explained",
            n_components,
            self._pca.explained_variance_ratio_.sum() * 100,
        )

    def transform(self, eeg_epochs: list[np.ndarray]) -> np.ndarray:
        """Transform epochs to PCA features.

        Args:
            eeg_epochs: List of (channels, time) arrays.

        Returns:
            (N, n_components) feature matrix.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._pca is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        flat = self._flatten_epochs(eeg_epochs, target_len=self._target_len)
        return self._pca.transform(flat).astype(np.float32)

    def fit_transform(self, eeg_epochs: list[np.ndarray]) -> np.ndarray:
        """Fit PCA and transform in one step."""
        self.fit(eeg_epochs)
        return self.transform(eeg_epochs)


class MeanPowerSpectrumBaseline:
    """Mean power spectral density across channels as features.

    Computes FFT of each epoch, averages the magnitude spectrum across
    channels, and uses the resulting frequency-domain vector as features.
    """

    def __init__(self, n_bins: int = 128) -> None:
        """Initialize with target feature dimensionality.

        Args:
            n_bins: Number of frequency bins to keep. Spectrum is
                linearly interpolated to this size.
        """
        self.n_bins = n_bins

    def extract(self, eeg_epochs: list[np.ndarray]) -> np.ndarray:
        """Compute mean power spectrum features for each epoch.

        Args:
            eeg_epochs: List of (channels, time) arrays.

        Returns:
            (N, n_bins) feature matrix (log power).
        """
        features = np.zeros((len(eeg_epochs), self.n_bins), dtype=np.float32)
        for i, epoch in enumerate(eeg_epochs):
            # Compute FFT magnitude for each channel
            fft_mag = np.abs(np.fft.rfft(epoch, axis=1))
            # Mean across channels
            mean_spectrum = fft_mag.mean(axis=0)
            # Log-transform (add small epsilon to avoid log(0))
            log_spectrum = np.log1p(mean_spectrum)
            # Interpolate to fixed size
            features[i] = np.interp(
                np.linspace(0, len(log_spectrum) - 1, self.n_bins),
                np.arange(len(log_spectrum)),
                log_spectrum,
            )

        return features
