"""Per-variable spectral + gradient statistics for anemoi datasets.

Self-contained numpy helpers used by the Anemoi target's
``calc_spectral_gradient_stats`` method to compute:

    * ``msh_beta``         — 1 / ⟨AMSE_j⟩, per variable
    * ``gradient_x_stdev`` — RMS of 4th-order finite-difference ∂/∂x
    * ``gradient_y_stdev`` — RMS of 4th-order finite-difference ∂/∂y

All operations are on 2D fields reshaped from the flattened ``cell`` axis
(``cell == H * W`` when ``do_flatten_grid=True``). The anemoi target
already flattens grids this way, so these helpers match that convention.

The AMSE proxy follows FastNet (arXiv:2509.17658, Eq. 3) with γ_k
wavenumber weighting and a coherence term, evaluated against zero-pred:

    AMSE_k(0, x) = (√0 − √PSD_k(x))² + 2·max(0, PSD_k(x)) · (1 − 0)
                 = PSD_k(x) + 2·PSD_k(x) = 3·PSD_k(x)

The factor 3 is variable-independent and cancels in β = 1/⟨AMSE⟩ ratios,
so it matches what the running loss's online Welford converges to at
initialization.

Numpy only — no torch dependency so these helpers can run inside any
ufs2arco build environment without extra conda pulls.
"""

from __future__ import annotations

import logging

import numpy as np

LOGGER = logging.getLogger("ufs2arco.targets.spectral_utils")

# 4th-order central-difference stencil, matching anemoi training's
# horizontal_gradient._CENTRAL_DIFF_4.
CENTRAL_DIFF_4 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0


# ----------------------------------------------------------------------
# Radial bin + conjugate-pair weight for 2D rfft layout
# ----------------------------------------------------------------------
def build_radial_bins(x_dim: int, y_dim: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(bin_idx, conjugate_weight, n_bins)`` for an rfft2 layout.

    Matches anemoi.training.losses.graphcast_msh._build_radial_bins.
    """
    kx = np.fft.fftfreq(x_dim) * x_dim
    ky = np.fft.rfftfreq(y_dim) * y_dim
    kxv, kyv = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kxv**2 + kyv**2)
    bin_idx = np.round(k_mag).astype(np.int64)

    # Non-DC-non-Nyquist rfft columns represent TWO modes in the full spectrum
    # (F and F*), so double-weight them so summed PSD matches |fft2|².
    weight = np.full_like(k_mag, 2.0)
    weight[:, 0] = 1.0
    if y_dim % 2 == 0:
        weight[:, -1] = 1.0

    n_bins = int(bin_idx.max()) + 1
    return bin_idx, weight, n_bins


def gamma_k_weights(
    n_bins: int,
    *,
    min_weight: float = 1.0,
    exponent: float = 3.0**0.5,
) -> np.ndarray:
    """FastNet γ_k = max(N_k · k^√3, 1.0) wavenumber weighting.

    Emphasises small-scale errors that the k^-3 / k^-5/3 atmospheric PSD
    decay would otherwise under-weight.
    """
    k = np.arange(n_bins, dtype=np.float64)
    raw = np.clip(k, 0.0, None) ** exponent
    nonzero = raw[raw > 0]
    if nonzero.size:
        raw = raw / max(float(nonzero.mean()), 1e-12)
    return np.clip(raw, min_weight, None)


# ----------------------------------------------------------------------
# Per-variable radial bin sum
# ----------------------------------------------------------------------
def radial_bin_sum_per_var(
    field_flat: np.ndarray,  # (V, n_pixels)
    bin_idx_flat: np.ndarray,  # (n_pixels,)
    n_bins: int,
) -> np.ndarray:
    """Scatter-sum ``field_flat`` into radial bins per variable.

    Returns (V, n_bins). Uses np.bincount per variable — fast and clear.
    """
    V = field_flat.shape[0]
    out = np.zeros((V, n_bins), dtype=field_flat.dtype)
    for v in range(V):
        out[v] = np.bincount(bin_idx_flat, weights=field_flat[v], minlength=n_bins)
    return out


# ----------------------------------------------------------------------
# Horizontal gradient via depth-wise 4th-order central diff + reflect pad
# ----------------------------------------------------------------------
def central_diff_x(field_2d: np.ndarray, kernel: np.ndarray = CENTRAL_DIFF_4) -> np.ndarray:
    """∂/∂x. ``field_2d`` shape (..., H, W). Reflect-padded stencil."""
    pad = (kernel.size - 1) // 2
    padded = np.pad(field_2d, [(0, 0)] * (field_2d.ndim - 1) + [(pad, pad)], mode="reflect")
    # Manual convolution: y[..., i] = Σ_k kernel[k] · padded[..., i+k]
    out = np.zeros_like(field_2d)
    for k in range(kernel.size):
        out += kernel[k] * padded[..., k : k + field_2d.shape[-1]]
    return out


def central_diff_y(field_2d: np.ndarray, kernel: np.ndarray = CENTRAL_DIFF_4) -> np.ndarray:
    """∂/∂y. ``field_2d`` shape (..., H, W). Reflect-padded stencil."""
    pad = (kernel.size - 1) // 2
    pad_spec = [(0, 0)] * (field_2d.ndim - 2) + [(pad, pad), (0, 0)]
    padded = np.pad(field_2d, pad_spec, mode="reflect")
    out = np.zeros_like(field_2d)
    for k in range(kernel.size):
        out += kernel[k] * padded[..., k : k + field_2d.shape[-2], :]
    return out


# ----------------------------------------------------------------------
# Per-variable AMSE
# ----------------------------------------------------------------------
def amse_per_var(
    field_2d: np.ndarray,  # (V, H, W)
    bin_idx_flat: np.ndarray,
    weight_flat: np.ndarray,
    gamma_k: np.ndarray,
    n_bins: int,
    coherence_weight: float = 1.0,
) -> np.ndarray:
    """AMSE_j(0, field) = Σ_k γ_k · AMSE_k where AMSE_k(0, x) = 3·PSD_k(x).

    Returns (V,) float64.
    """
    V, H, W = field_2d.shape
    alpha = np.fft.rfft2(field_2d.astype(np.float32), axes=(-2, -1))  # (V, H, W//2+1)
    power = alpha.real.astype(np.float64) ** 2 + alpha.imag.astype(np.float64) ** 2

    W_r = alpha.shape[-1]
    n_pix = H * W_r
    power_flat = (power.reshape(V, n_pix) * weight_flat[None, :])  # conjugate-pair weight

    psd = radial_bin_sum_per_var(power_flat, bin_idx_flat, n_bins)  # (V, n_bins)

    # AMSE_k(0, x) with coherence term:
    #   amp_err = (√0 − √PSD_k)² = PSD_k
    #   coh_err = 2 · max(0, PSD_k) · (1 − 0) = 2 · PSD_k
    amse_k = psd + coherence_weight * 2.0 * psd  # (V, n_bins)
    amse_k = amse_k * gamma_k[None, :]
    return amse_k.sum(axis=-1)  # (V,)


# ----------------------------------------------------------------------
# One-frame contribution to all 6 accumulators
# ----------------------------------------------------------------------
def accumulate_frame(
    tendency_phys: np.ndarray,   # (V, H, W) physical tendency x_t − x_{t-1}
    tend_stdev_safe: np.ndarray,  # (V,) clamped to min 1e-6
    bin_idx_flat: np.ndarray,
    weight_flat: np.ndarray,
    gamma_k: np.ndarray,
    n_bins: int,
    accumulators: dict,           # mutated in place; see build_accumulators
    coherence_weight: float = 1.0,
) -> int:
    """Add this frame's contribution to the 6 per-variable accumulators.

    Per-variable NaN handling: variables whose tendency field has any NaN in
    this frame are EXCLUDED from this frame's accumulation (their per-variable
    count isn't bumped). Valid variables contribute normally. This matches
    the behavior of anemoi-datasets' skipna=True aggregations.

    Returns the number of valid variables contributed by this frame (0 if
    the frame is all-NaN).
    """
    V, H, W = tendency_phys.shape

    # Per-variable validity: True if variable has NO NaN anywhere in this frame
    valid_mask = ~np.isnan(tendency_phys).any(axis=(-2, -1))  # (V,)
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return 0

    # Replace NaN with 0 so the FFT + stencil don't propagate them.
    # We mask the per-variable output with valid_mask before accumulating,
    # so NaN-variables contribute nothing.
    tendency_phys_clean = np.nan_to_num(tendency_phys, nan=0.0)
    tendency_res_clean = tendency_phys_clean / tend_stdev_safe[:, None, None]

    # β_j (AMSE) — mask per-variable output
    amse_phys = amse_per_var(tendency_phys_clean, bin_idx_flat, weight_flat, gamma_k, n_bins, coherence_weight)
    amse_res = amse_per_var(tendency_res_clean, bin_idx_flat, weight_flat, gamma_k, n_bins, coherence_weight)
    accumulators["amse_phys_sum"] += amse_phys * valid_mask
    accumulators["amse_res_sum"] += amse_res * valid_mask
    accumulators["amse_count_per_var"] += valid_mask.astype(np.float64)

    # σ_∂x / σ_∂y: per-variable sum of squares over pixels
    dx_phys = central_diff_x(tendency_phys_clean)
    dy_phys = central_diff_y(tendency_phys_clean)
    dx_res = central_diff_x(tendency_res_clean)
    dy_res = central_diff_y(tendency_res_clean)
    accumulators["dx_sq_phys_sum"] += ((dx_phys.astype(np.float64) ** 2).sum(axis=(-2, -1))) * valid_mask
    accumulators["dy_sq_phys_sum"] += ((dy_phys.astype(np.float64) ** 2).sum(axis=(-2, -1))) * valid_mask
    accumulators["dx_sq_res_sum"] += ((dx_res.astype(np.float64) ** 2).sum(axis=(-2, -1))) * valid_mask
    accumulators["dy_sq_res_sum"] += ((dy_res.astype(np.float64) ** 2).sum(axis=(-2, -1))) * valid_mask
    accumulators["grad_pixel_count_per_var"] += valid_mask.astype(np.float64) * (H * W)

    return n_valid


def build_accumulators(n_vars: int) -> dict:
    """Allocate the per-variable fp64 accumulators expected by ``accumulate_frame``.

    All accumulators are per-variable (shape (V,)). Counts are per-variable
    too so that sparse-NaN variables are handled correctly: a variable that
    appears valid in half the frames gets its stats computed over that half,
    independent of other variables.
    """
    return {
        "amse_phys_sum": np.zeros(n_vars, dtype=np.float64),
        "amse_res_sum": np.zeros(n_vars, dtype=np.float64),
        "amse_count_per_var": np.zeros(n_vars, dtype=np.float64),
        "dx_sq_phys_sum": np.zeros(n_vars, dtype=np.float64),
        "dy_sq_phys_sum": np.zeros(n_vars, dtype=np.float64),
        "dx_sq_res_sum": np.zeros(n_vars, dtype=np.float64),
        "dy_sq_res_sum": np.zeros(n_vars, dtype=np.float64),
        "grad_pixel_count_per_var": np.zeros(n_vars, dtype=np.float64),
    }


def finalize_stats(
    accumulators: dict,
    *,
    beta_floor: float = 1e-10,
    sigma_floor: float = 0.0,
) -> dict[str, np.ndarray]:
    """Turn summed accumulators into final β (= 1/⟨AMSE⟩) and σ arrays.

    Returns a dict with keys:
        msh_beta_physical
        msh_beta_residual
        gradient_x_stdev_physical
        gradient_y_stdev_physical
        gradient_x_stdev_residual
        gradient_y_stdev_residual
    """
    # Per-variable safe divisors (at least 1 to avoid div-by-zero for vars
    # with 0 valid frames — their final β will saturate at the ceiling).
    n_amse = np.maximum(accumulators["amse_count_per_var"], 1.0)
    n_pix = np.maximum(accumulators["grad_pixel_count_per_var"], 1.0)

    mean_amse_phys = accumulators["amse_phys_sum"] / n_amse
    mean_amse_res = accumulators["amse_res_sum"] / n_amse
    beta_phys = 1.0 / np.clip(mean_amse_phys, beta_floor, None)
    beta_res = 1.0 / np.clip(mean_amse_res, beta_floor, None)

    sx_phys = np.sqrt(np.clip(accumulators["dx_sq_phys_sum"] / n_pix, sigma_floor, None))
    sy_phys = np.sqrt(np.clip(accumulators["dy_sq_phys_sum"] / n_pix, sigma_floor, None))
    sx_res = np.sqrt(np.clip(accumulators["dx_sq_res_sum"] / n_pix, sigma_floor, None))
    sy_res = np.sqrt(np.clip(accumulators["dy_sq_res_sum"] / n_pix, sigma_floor, None))

    return {
        "msh_beta_physical": beta_phys.astype(np.float64),
        "msh_beta_residual": beta_res.astype(np.float64),
        "gradient_x_stdev_physical": sx_phys.astype(np.float64),
        "gradient_y_stdev_physical": sy_phys.astype(np.float64),
        "gradient_x_stdev_residual": sx_res.astype(np.float64),
        "gradient_y_stdev_residual": sy_res.astype(np.float64),
    }
