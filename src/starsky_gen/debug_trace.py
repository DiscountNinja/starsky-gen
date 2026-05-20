"""Session debug tracing (NDJSON append)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SESSION = os.environ.get("STARSKY_DEBUG_SESSION", "408793")
_DEBUG_LOG = Path(
    os.environ.get(
        "STARSKY_DEBUG_LOG",
        str(_REPO_ROOT / ".cursor" / f"debug-{_SESSION}.log"),
    )
)


def debug_log(
    location: str,
    message: str,
    data: dict[str, Any] | None = None,
    *,
    hypothesis_id: str = "",
    run_id: str = "pre-fix",
) -> None:
    # #region agent log
    payload = {
        "sessionId": _SESSION,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(time.time() * 1000),
        "logPath": str(_DEBUG_LOG),
    }
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=float) + "\n")
    except OSError as exc:
        import sys

        print(f"[starsky debug] log write failed: {_DEBUG_LOG}: {exc}", file=sys.stderr)
    # #endregion


def field_stats(arr, mask=None) -> dict[str, float]:
    import numpy as np

    x = np.asarray(arr, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape[:2]:
            return {}
        if x.ndim == 3:
            x = x[m]
        else:
            x = x[m]
    if x.size == 0:
        return {"n": 0.0}
    return {
        "n": float(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.percentile(x, 50)),
        "p92": float(np.percentile(x, 92)),
    }


def field_histogram_band(
    arr,
    mask,
    *,
    lo_edge: float,
    hi_edge: float,
) -> dict[str, float]:
    """Fraction of band pixels pinned near floor/ceiling (mean alone hides bimodal slabs)."""
    import numpy as np

    x = np.asarray(arr, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != x.shape[:2]:
            return {}
        x = x[m] if x.ndim == 2 else x[m]
    if x.size == 0:
        return {}
    span = float(np.max(x) - np.min(x)) + 1e-8
    near_lo = float(np.mean(x <= lo_edge))
    near_hi = float(np.mean(x >= hi_edge))
    return {
        "frac_near_lo": near_lo,
        "frac_near_hi": near_hi,
        "dynamic_range": span,
    }


def highpass_band_stats(
    arr,
    mask=None,
    *,
    sigma_frac: float = 0.008,
) -> dict[str, float]:
    """High-pass std — tiny puff/cloud breakup (low = smooth wash)."""
    import numpy as np

    from starsky_gen.procedural_noise import gaussian_blur_pil

    x = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    if x.ndim == 3:
        x = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    h, w = x.shape[:2]
    sig = float(np.clip(max(h, w) * float(sigma_frac), 0.45, 12.0))
    hp = x - gaussian_blur_pil(x, sig, periodic_x=True)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == x.shape:
            hp = hp[m]
    if hp.size == 0:
        return {"hp_std": 0.0}
    return {"hp_std": float(np.std(hp))}


def puff_peak_count(
    arr,
    mask=None,
    *,
    thresh: float = 0.62,
) -> dict[str, float]:
    """Rough count of local puff peaks (higher = more discrete cotton-ball cells)."""
    import numpy as np

    x = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
    if x.ndim == 3:
        x = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    peaks = x[1:-1, 1:-1] >= thresh
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        peaks &= x[1 + dy : x.shape[0] - 1 + dy, 1 + dx : x.shape[1] - 1 + dx] <= x[1:-1, 1:-1]
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == x.shape:
            peaks &= m[1:-1, 1:-1]
    return {"puff_peaks": float(np.count_nonzero(peaks))}


def band_absorption_extinction_coupling(
    absorption: np.ndarray,
    transmission: np.ndarray,
    mask=None,
) -> dict[str, float]:
    """How well dust absorption (high=dark) tracks opacity (low T). corr(abs, 1-T) should be > 0."""
    import numpy as np

    from starsky_gen.procedural_noise import gaussian_blur_pil

    a = np.clip(np.asarray(absorption, dtype=np.float64), 0.0, 1.0)
    t = np.clip(np.asarray(transmission, dtype=np.float64), 0.0, 1.0)
    op = 1.0 - t
    m = None
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == a.shape:
            a_s, op_s = a[m], op[m]
        else:
            a_s, op_s = a.ravel(), op.ravel()
    else:
        a_s, op_s = a.ravel(), op.ravel()
    if a_s.size < 64:
        return {"abs_opacity_corr": 0.0, "abs_hp_opacity_corr": 0.0, "opacity_p50": 0.0, "trans_p50": 0.0}
    c = float(np.corrcoef(a_s, op_s)[0, 1])
    sig = float(np.clip(max(a.shape) * 0.008, 0.45, 12.0))
    a_hp = a - gaussian_blur_pil(a, sig, periodic_x=True)
    op_hp = op - gaussian_blur_pil(op, sig, periodic_x=True)
    if m is not None and m.shape == a.shape:
        a_hp_s, op_hp_s = a_hp[m], op_hp[m]
    else:
        a_hp_s, op_hp_s = a_hp.ravel(), op_hp.ravel()
    ch = float(np.corrcoef(a_hp_s, op_hp_s)[0, 1]) if a_hp_s.size > 64 else 0.0
    return {
        "abs_opacity_corr": c,
        "abs_hp_opacity_corr": ch,
        "opacity_p50": float(np.percentile(op_s, 50)),
        "trans_p50": float(np.percentile(1.0 - op_s, 50)),
    }


def gradient_anisotropy(arr, mask=None) -> dict[str, float]:
    """gy/gx >> 1 suggests vertical comb/striation in equirect band."""
    import numpy as np

    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 3:
        x = x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    gx = np.abs(np.diff(x, axis=1))
    gy = np.abs(np.diff(x, axis=0))
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == x.shape:
            gx = gx * m[:, :-1]
            gy = gy * m[:-1, :]
    gxm = float(np.mean(gx))
    gym = float(np.mean(gy))
    ratio = gym / (gxm + 1e-8)
    return {
        "gx_mean": gxm,
        "gy_mean": gym,
        "gy_over_gx": ratio,
        "vert_dominant": float(ratio > 1.35),
    }
