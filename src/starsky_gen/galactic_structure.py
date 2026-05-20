"""Unified galactic morphology: one latent stack → stars, dust, unresolved, PSF.

Morphology smoothing pipeline (debug layers show puffy/turbulent HF; final render can wash it):
  - dust_absorption_morph / dust_transmission_morph are frozen at build; use these for structure.
  - merge_nebula_extinction: never max(morph, 1-T) alone — when T≈void_floor that slabs effective
    absorption (~0.9+) and erases puff detail in dust_A_effective exports.
  - build_unresolved_intensity_from_master: u∝g×d damps the band when T is low; inject turb/dust-open HF.
  - compose_inherited_unresolved_field: continuum Gaussian + speckle σ-pyramid blurs deposits; keep a
    morph-HF speckle tier and lower σ caps so tight clouds survive before extinction.
  - generator post-pass: extinction on canvas, haze/cloud body, band highlight compress, and
    harmonize_diffuse_canvas_chroma each low-pass the band — morph-primary paths use HF guides.
  - density_G is intentionally smoother (star placement); turbulent ISM detail lives in dust/ISM maps.
  Runtime: debug logs hypothesisId X (hp_std at merge / unresolved / extinction-on-canvas).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from starsky_gen.dust_field import (
    build_band_disruption_field,
    build_filament_erosion_map,
    carve_dust_transmission_microstructure,
    enrich_dust_absorption_microstructure,
    transmission_from_absorption_map,
)
from starsky_gen.placement import (
    build_cluster_density_modulation,
    build_equirect_density_map,
    build_master_density_field,
    pick_association_peaks,
    sample_lon_lat_poisson,
)


def _local_seed_epsilon_field(
    rng: np.random.Generator,
    height: int,
    width: int,
    *,
    amplitude: float,
    base_scale: float = 0.11,
    octaves: int = 3,
    periodic_x: bool = True,
) -> np.ndarray:
    """Correlated zero-mean ε so seed shifts morphology locally, not only architecture."""
    amp = float(amplitude)
    if amp < 1e-8:
        return np.zeros((height, width), dtype=np.float64)
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d

    ch, cw = max(4, height // 40), max(6, width // 40)
    n = fbm2d(rng, ch, cw, base_scale=base_scale, octaves=octaves, periodic_x=periodic_x)
    n = _resize_bilinear(n, height, width, periodic_x=periodic_x)
    n = (n - float(np.mean(n))) / (float(np.std(n)) + 1e-8)
    return (n * amp).astype(np.float64)


def _spawn_morphology_eps_rng(parent: np.random.Generator, tag: int) -> np.random.Generator:
    return np.random.default_rng(int(parent.integers(0, 2**31)) ^ int(tag))


def _apply_multiplicative_epsilon(
    field: np.ndarray,
    epsilon: np.ndarray,
    *,
    renorm: bool = True,
    floor: float = 1e-4,
    ceiling: float = 1.0,
) -> np.ndarray:
    out = np.clip(np.asarray(field, dtype=np.float64) * (1.0 + epsilon), floor, ceiling)
    if renorm:
        mx = float(np.max(out))
        if mx > 1e-8:
            out = out / mx
    return out.astype(np.float64)


def _downsample_field(field: np.ndarray, sh: int, sw: int) -> np.ndarray:
    """Block-average downsample to (sh, sw)."""
    h, w = field.shape
    sh = max(1, min(sh, h))
    sw = max(1, min(sw, w))
    bh, bw = h // sh, w // sw
    if bh < 1 or bw < 1:
        from starsky_gen.procedural_noise import _resize_bilinear

        return _resize_bilinear(field, sh, sw, periodic_x=True)
    trimmed = field[: bh * sh, : bw * sw]
    return trimmed.reshape(sh, bh, sw, bw).mean(axis=(1, 3))


@dataclass
class MorphologyExtinctionMaps:
    """Precomputed absorption/disruption maps for nebula extinction path."""

    erosion: np.ndarray
    fractal_dark: np.ndarray
    disruption: np.ndarray


@dataclass
class GalacticMorphology:
    """Latent galactic maps (H×W); all populations inherit from one morphology."""

    width: int
    height: int
    latent_ridge: np.ndarray
    latent_turb: np.ndarray
    stellar_density: np.ndarray
    star_formation: np.ndarray
    dust_absorption: np.ndarray
    dust_transmission: np.ndarray
    dust_absorption_morph: np.ndarray
    dust_transmission_morph: np.ndarray
    cluster_prob: np.ndarray
    unresolved_prior: np.ndarray
    unresolved_coarse: np.ndarray
    unresolved_mid: np.ndarray
    resolve_weight: np.ndarray
    void_mask: np.ndarray
    obliteration_mask: np.ndarray
    brutal_erasure_mask: np.ndarray
    structure_survival: np.ndarray
    vertical_extent: np.ndarray
    lat_structure_shift: np.ndarray
    lon_asymmetry: np.ndarray
    gold_population_weight: np.ndarray
    psf_environment: np.ndarray
    disk_weight: np.ndarray
    unresolved_accum: np.ndarray = field(repr=False)
    disk_thickness_modulation: np.ndarray | None = None
    extinction_maps: MorphologyExtinctionMaps | None = None

    # Back-compat aliases
    @property
    def stellar_density_G(self) -> np.ndarray:
        return self.stellar_density

    def merge_nebula_extinction(self, extinction: np.ndarray) -> None:
        ext = np.asarray(extinction, dtype=np.float64)
        if ext.shape != self.dust_transmission.shape:
            return
        self.dust_transmission[:] = np.minimum(self.dust_transmission, ext)
        # Keep morph puff/lane HF in absorption; max(morph, 1-T) flattened effective maps
        # when T≈void_floor (slab ~0.9+ absorption erased turbulent detail in exports/coupling).
        morph = np.clip(self.dust_absorption_morph, 0.0, 1.0)
        ext_abs = np.clip(1.0 - self.dust_transmission, 0.0, 1.0)
        from starsky_gen.procedural_noise import gaussian_blur_pil

        sig = float(np.clip(max(morph.shape) * 0.010, 0.8, 12.0))
        hp_m = np.clip(morph - gaussian_blur_pil(morph, sig, periodic_x=True) * 0.76, -0.4, 0.4)
        hp_e = np.clip(ext_abs - gaussian_blur_pil(ext_abs, sig, periodic_x=True) * 0.76, -0.4, 0.4)
        blended = np.clip(morph * 0.58 + ext_abs * 0.42, 0.0, 1.0)
        self.dust_absorption[:] = np.clip(blended + hp_m * 0.30 - hp_e * 0.06, 0.06, 0.94)

    def deposit_unresolved(self, yi: int, xi: int, flux: float) -> None:
        if flux <= 1e-12:
            return
        yi = int(np.clip(yi, 0, self.height - 1))
        xi = int(xi % self.width)
        atten = float(self.dust_transmission[yi, xi])
        oblit = float(self.obliteration_mask[yi, xi])
        self.unresolved_accum[yi, xi] += float(flux) * (0.35 + 0.65 * atten) * (1.0 + 1.15 * oblit)

    def unresolved_intensity(self) -> np.ndarray:
        """Master-density inherited unresolved field (prior + catalog deposits)."""
        return np.clip(self.unresolved_prior + self.unresolved_accum, 0.0, None)

    def unresolved_total(self) -> np.ndarray:
        return self.unresolved_intensity()

    def unresolved_speckle_rate(self) -> np.ndarray:
        """Speckle rate high where luminous unresolved dominates over resolve exceptions."""
        u = self.unresolved_intensity()
        u = u / (float(np.percentile(u, 99.0)) + 1e-8)
        w = np.clip(1.0 - self.resolve_weight, 0.06, 1.0)
        return np.clip(u * w * self.stellar_density, 0.0, None).astype(np.float64)

    def sample_stellar(self, yi: int, xi: int) -> float:
        return float(self.stellar_density[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_dust(self, yi: int, xi: int) -> float:
        return float(self.dust_transmission[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_psf_env(self, yi: int, xi: int) -> float:
        return float(self.psf_environment[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_resolve_weight(self, yi: int, xi: int) -> float:
        return float(self.resolve_weight[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_obliteration(self, yi: int, xi: int) -> float:
        return float(self.obliteration_mask[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_brutal_erasure(self, yi: int, xi: int) -> float:
        return float(self.brutal_erasure_mask[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def sample_gold_population(self, yi: int, xi: int) -> float:
        return float(self.gold_population_weight[int(np.clip(yi, 0, self.height - 1)), int(xi) % self.width])

    def derived_maps(self) -> dict[str, np.ndarray]:
        return {
            "stellar_density": self.stellar_density,
            "star_formation": self.star_formation,
            "dust_absorption": self.dust_absorption,
            "dust_transmission": self.dust_transmission,
            "dust_absorption_morph": self.dust_absorption_morph,
            "dust_transmission_morph": self.dust_transmission_morph,
            "cluster_prob": self.cluster_prob,
            "unresolved_total": self.unresolved_total(),
            "resolve_weight": self.resolve_weight,
            "void_mask": self.void_mask,
            "obliteration_mask": self.obliteration_mask,
            "brutal_erasure_mask": self.brutal_erasure_mask,
            "structure_survival": self.structure_survival,
            "vertical_extent": self.vertical_extent,
            "lon_asymmetry": self.lon_asymmetry,
            "gold_population_weight": self.gold_population_weight,
            "psf_environment": self.psf_environment,
            "disk_weight": self.disk_weight,
            "disk_thickness_modulation": self.disk_thickness_modulation
            if self.disk_thickness_modulation is not None
            else np.zeros_like(self.disk_weight),
            "latent_ridge": self.latent_ridge,
        }


# Backward compatibility
GalacticStructure = GalacticMorphology


def build_fractal_from_latent_ridge(
    latent_ridge: np.ndarray,
    height: int,
    width: int,
    *,
    erosion_power: float = 1.72,
) -> np.ndarray:
    """Fractal absorption from shared ridge latent (no new RNG filament pass)."""
    ridge = np.asarray(latent_ridge, dtype=np.float64)
    if ridge.shape != (height, width):
        from starsky_gen.procedural_noise import _resize_bilinear

        ridge = _resize_bilinear(ridge, height, width, periodic_x=True)
    mf = np.clip((1.0 - np.abs(ridge * 2.0 - 1.0)) ** float(erosion_power), 0.0, 1.0)
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float64)[:, None]
    plane = np.exp(-((yy**2) / 0.45))
    return np.clip(mf * plane * 0.55, 0.0, 1.0).astype(np.float64)


def local_field_variance_stretch(
    field: np.ndarray,
    *,
    variance: float = 1.32,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> np.ndarray:
    """Mean-preserving contrast stretch (local variance ↑, not global mean dust ↑)."""
    x = np.clip(np.asarray(field, dtype=np.float64), floor, ceiling)
    v = float(np.clip(variance, 0.5, 2.5))
    if abs(v - 1.0) < 1e-6:
        return x
    mu = float(np.mean(x))
    return np.clip(mu + (x - mu) * v, floor, ceiling).astype(np.float64)


def band_gated_variance_stretch(
    field: np.ndarray,
    band_weight: np.ndarray,
    *,
    variance: float = 1.32,
    floor: float = 0.0,
    ceiling: float = 1.0,
    halo_ceiling: float = 0.38,
) -> np.ndarray:
    """Variance stretch in the disk only; halo stays smooth (no pole boiling)."""
    from starsky_gen.dust_field import _band_host_gate

    x = np.clip(np.asarray(field, dtype=np.float64), floor, ceiling)
    bg = _band_host_gate(band_weight, x.shape)
    stretched = local_field_variance_stretch(
        x, variance=variance, floor=floor, ceiling=ceiling
    )
    out = x.copy()
    core = bg > 0.22
    if bool(np.any(core)):
        out[core] = stretched[core]
    halo = bg < 0.26
    if bool(np.any(halo)):
        out[halo] = np.clip(x[halo] * 0.86, floor, float(halo_ceiling))
    return out.astype(np.float64)


def build_obliteration_mask(
    stellar_density: np.ndarray,
    dust_absorption: np.ndarray,
    void_mask: np.ndarray,
    disk_weight: np.ndarray,
    *,
    strength: float = 0.82,
) -> np.ndarray:
    """Dense stellar regions behind thick dust → few resolved survivors, many neighbors bright."""
    from starsky_gen.procedural_noise import gaussian_blur_pil

    g = np.clip(np.asarray(stellar_density, dtype=np.float64), 0.0, 1.0)
    dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
    void_w = np.clip(np.asarray(void_mask, dtype=np.float64), 0.0, 1.0)
    dw = np.clip(np.asarray(disk_weight, dtype=np.float64), 0.0, 1.0)
    scale = float(max(g.shape))
    sig = float(np.clip(scale * 0.045, 6.0, 52.0))
    g_hp = np.clip(g - gaussian_blur_pil(g, sig, periodic_x=True) * 0.84, 0.0, 1.0)
    dust_hp = np.clip(dust - gaussian_blur_pil(dust, sig, periodic_x=True) * 0.84, 0.0, 1.0)
    g_lo = float(np.percentile(g_hp, 58.0))
    g_hi = float(np.percentile(g_hp, 88.0))
    dense = np.clip((g_hp - g_lo) / max(g_hi - g_lo, 1e-8), 0.0, 1.0) ** 1.15
    dust_lo = float(np.percentile(dust_hp, 52.0))
    dust_hi = float(np.percentile(dust_hp, 86.0))
    dusty = np.clip((dust_hp - dust_lo) / max(dust_hi - dust_lo, 1e-8), 0.0, 1.0) ** 1.22
    oblit = np.clip(dense * dusty * (0.48 + 0.52 * dw) * (1.0 + 0.28 * void_w), 0.0, 1.0)
    s = float(np.clip(strength, 0.0, 1.25))
    return np.clip(oblit**1.08 * s, 0.0, 1.0).astype(np.float64)


def build_galactic_morphology(
    width: int,
    height: int,
    rng: np.random.Generator,
    *,
    disk_height: float = 0.19,
    halo_fraction: float = 0.22,
    halo_power: float = 1.35,
    band_lat_sigma: float = 0.10,
    band_rotation_deg: float = 2.5,
    band_curvature_amp: float = 0.04,
    band_thickness_asymmetry: float = 0.38,
    disk_mesoscale_thickness_strength: float = 0.58,
    cluster_strength: float = 0.72,
    turbulence_strength: float = 0.85,
    void_strength: float = 0.72,
    scar_strength: float = 0.68,
    placement_asymmetry: float = 0.0,
    drop_strength: float = 0.54,
    discontinuity_cut_strength: float = 0.55,
    macro_void_count: int = 3,
    periodic_x: bool = True,
    seed_perturb_scale: float = 1.0,
    sf_perturb: float = 0.16,
    dust_perturb: float = 0.13,
    dust_micro_strength: float = 1.25,
    cluster_perturb: float = 0.22,
    local_variance: float = 1.32,
    obliteration_strength: float = 0.82,
    regional_chaos: float = 0.38,
    generation_phase: float = 0.0,
    vertical_extent_strength: float = 0.72,
    structure_host_latitude_scale: float = 1.85,
    longitude_asymmetry_strength: float = 0.88,
    brutal_erasure_strength: float = 0.78,
    seam_guard_strength: float = 0.85,
    disaster_peak_count: int = 3,
    gold_population_patchiness: float = 1.28,
) -> GalacticMorphology:
    """Build unified morphology from one procedural seed lineage."""
    from starsky_gen.dust_field import _blur_x_only_field, _blur_y_only_field
    from starsky_gen.procedural_noise import _resize_bilinear, fbm2d, ridged_fbm2d

    ch, cw = max(12, height // 28), max(16, width // 38)
    latent_ridge_lo = ridged_fbm2d(
        rng,
        ch,
        cw,
        base_scale=0.08,
        octaves=4,
        periodic_x=periodic_x,
        elongate_along_x=2.1,
    )
    latent_turb_lo = fbm2d(rng, ch, cw, base_scale=0.10, octaves=3, periodic_x=periodic_x)
    latent_ridge_hi = _resize_bilinear(latent_ridge_lo, height, width, periodic_x=periodic_x)
    latent_turb = _resize_bilinear(latent_turb_lo, height, width, periodic_x=periodic_x)
    latent_ridge = _blur_x_only_field(latent_ridge_hi, 2.4, periodic_x=periodic_x)
    latent_ridge = _blur_y_only_field(latent_ridge, 0.55)
    if max(height, width) >= 512:
        from starsky_gen.dust_field import attenuate_column_comb

        latent_ridge = attenuate_column_comb(
            latent_ridge, None, strength=0.48, periodic_x=periodic_x
        )
    cluster_map = build_cluster_density_modulation(
        height, width, rng, strength=cluster_strength, periodic_x=periodic_x
    )
    c = np.clip(cluster_map, 0.0, 1.0).astype(np.float64)
    perturb_s = float(np.clip(seed_perturb_scale, 0.0, 2.0))
    if perturb_s > 1e-6:
        eps_c = _local_seed_epsilon_field(
            _spawn_morphology_eps_rng(rng, 0xC1A5),
            height,
            width,
            amplitude=float(cluster_perturb) * perturb_s,
            base_scale=0.13,
            periodic_x=periodic_x,
        )
        c = _apply_multiplicative_epsilon(c, eps_c, renorm=True)

    g = build_master_density_field(
        width,
        height,
        rng,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        turbulence_strength=turbulence_strength,
        periodic_x=periodic_x,
    )
    sf = np.clip(g**1.4 * (0.55 + 0.45 * c), 0.0, 1.0)
    sf /= float(np.max(sf)) + 1e-8
    if perturb_s > 1e-6:
        eps_sf = _local_seed_epsilon_field(
            _spawn_morphology_eps_rng(rng, 0x5F00),
            height,
            width,
            amplitude=float(sf_perturb) * perturb_s,
            base_scale=0.09,
            octaves=4,
            periodic_x=periodic_x,
        )
        sf = _apply_multiplicative_epsilon(sf, eps_sf, renorm=True)

    band_pre = build_equirect_density_map(
        width,
        height,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        cluster_map=None,
    )
    band_w_pre = np.clip(band_pre / (float(np.max(band_pre)) + 1e-8), 0.0, 1.0)
    from starsky_gen.structure_envelope import apply_variable_band_thickness

    band_w_pre, _, _meso_pre = apply_variable_band_thickness(
        band_w_pre,
        rng,
        band_lat_sigma=band_lat_sigma,
        jitter_strength=0.92,
        band_curvature_amp=band_curvature_amp,
        thickness_asymmetry=band_thickness_asymmetry,
        mesoscale_strength=disk_mesoscale_thickness_strength,
        periodic_x=periodic_x,
    )

    ridge_dark = np.clip(1.0 - np.abs(latent_ridge * 2.0 - 1.0), 0.0, 1.0) ** 1.35
    scar = np.clip(ridge_dark**1.2 * float(scar_strength), 0.0, 1.0)
    dust_a = np.clip(0.06 + 0.78 * scar + 0.38 * (1.0 - g), 0.05, 1.0)
    lo_a = float(np.percentile(dust_a, 4.0))
    hi_a = float(np.percentile(dust_a, 90.0))
    if hi_a > lo_a + 1e-8:
        dust_a = np.clip((dust_a - lo_a) / (hi_a - lo_a), 0.0, 1.0)
    dust_a = np.clip(dust_a**1.08, 0.06, 1.0)
    lv = float(np.clip(local_variance, 0.5, 2.2))
    dust_a = band_gated_variance_stretch(
        dust_a, band_w_pre, variance=lv, floor=0.04, ceiling=1.0, halo_ceiling=0.46
    )
    from starsky_gen.dust_field import flatten_offband_dust_absorption

    dust_a = flatten_offband_dust_absorption(dust_a, band_w_pre, ceiling=0.44, soften=0.80)
    sf = local_field_variance_stretch(sf, variance=lv * 0.92, floor=0.0, ceiling=1.0)
    rc = float(np.clip(regional_chaos, 0.0, 0.85))
    if rc > 1e-6:
        phase = float(generation_phase)
        reg = np.clip(
            latent_turb * (0.82 + 0.18 * np.sin(2.0 * np.pi * (latent_turb * 1.7 + phase))),
            0.0,
            1.0,
        )
        mod = 1.0 + rc * (0.42 * reg + 0.28 * (latent_ridge - 0.5))
        sf = np.clip(sf * mod, 0.0, 1.0)
        from starsky_gen.dust_field import _band_host_gate

        chaos_gate = _band_host_gate(band_w_pre, (height, width))
        dust_a = np.clip(dust_a * (1.0 + 0.32 * rc * mod * chaos_gate), 0.04, 1.0)
    if perturb_s > 1e-6:
        eps_dust = _local_seed_epsilon_field(
            _spawn_morphology_eps_rng(rng, 0xD057),
            height,
            width,
            amplitude=float(dust_perturb) * perturb_s,
            base_scale=0.14,
            octaves=4,
            periodic_x=periodic_x,
        )
        dust_a = _apply_multiplicative_epsilon(
            dust_a, eps_dust, renorm=True, floor=0.05, ceiling=1.0
        )
    micro_s = float(np.clip(dust_micro_strength, 0.0, 1.5))
    if micro_s > 1e-6:
        from starsky_gen.dust_field import (
            _band_host_gate,
            enrich_dust_absorption_microstructure,
            remap_band_absorption_contrast,
        )

        band_core = _band_host_gate(band_w_pre, (height, width)) > 0.22
        if bool(np.any(band_core)):
            sub = dust_a[band_core]
            cap = float(np.percentile(sub, 84.0))
            if cap > 0.18:
                dust_a = dust_a.copy()
                dust_a[band_core] = np.clip(sub * (0.62 / cap), 0.08, 0.68)
        dust_a = enrich_dust_absorption_microstructure(
            dust_a,
            rng,
            latent_ridge_hi,
            latent_turb,
            periodic_x=periodic_x,
            strength=micro_s,
            band_weight=band_w_pre,
        )
        dust_a = remap_band_absorption_contrast(
            dust_a,
            band_w_pre,
            floor=0.10,
            span=0.68,
            p_lo=10.0,
            p_hi=84.0,
        )
    d = transmission_from_absorption_map(dust_a, void_floor=0.06, sharpness=2.88)
    if micro_s > 1e-6:
        from starsky_gen.dust_field import _band_host_gate

        band_t = _band_host_gate(band_w_pre, (height, width)) > 0.20
        if bool(np.any(band_t)):
            d_soft = transmission_from_absorption_map(
                dust_a, void_floor=0.08, sharpness=2.05
            )
            d = np.where(band_t, np.minimum(d, d_soft), d)
        d = carve_dust_transmission_microstructure(
            d,
            dust_a,
            periodic_x=periodic_x,
            strength=micro_s,
            void_floor=0.06,
            band_weight=band_w_pre,
        )

    void_mask = np.clip(ridge_dark**1.55 * float(void_strength), 0.0, 0.92)
    g = g * (1.0 - void_mask)
    g = _apply_macro_voids(
        g,
        rng,
        count=macro_void_count,
        height=height,
        width=width,
        strength=float(void_strength),
        periodic_x=periodic_x,
    )

    if float(placement_asymmetry) > 1e-5:
        xx = np.linspace(0.0, 1.0, width, dtype=np.float64)[None, :]
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        lon_warp = float(placement_asymmetry) * 0.12 * np.sin(2.0 * np.pi * xx * 2.3 + phase)
        g = g * (1.0 + lon_warp * (np.linspace(-1, 1, height)[:, None] * 0.5 + 0.5))

    rows, cols = pick_association_peaks(g * c, rng, n_peaks=12, min_sep_px=32.0)
    spike = np.zeros((height, width), dtype=np.float64)
    for r, c0 in zip(rows, cols, strict=False):
        ri, ci = int(np.clip(int(r), 0, height - 1)), int(c0) % width
        spike[ri, ci] = max(spike[ri, ci], float(rng.uniform(0.35, 0.85)))
    from starsky_gen.procedural_noise import gaussian_blur_pil

    from starsky_gen.structure_envelope import (
        build_brutal_erasure_mask,
        build_gold_population_field,
        build_longitude_asymmetry,
        build_vertical_structure_envelope,
        scatter_disaster_lon_peaks,
        seam_safe_lon_weights,
    )

    seam_w = seam_safe_lon_weights(width, guard_frac=0.055)
    spike = spike * np.clip(seam_w ** float(np.clip(seam_guard_strength, 0.0, 1.2)), 0.12, 1.0)
    spike = gaussian_blur_pil(spike, 1.2, periodic_x=periodic_x)
    g = np.clip(g * (1.0 + spike * 0.95), 0.0, 1.0)
    lon_asym = build_longitude_asymmetry(width, strength=longitude_asymmetry_strength)
    g = np.clip(g * lon_asym, 0.0, 1.0)
    sf = np.clip(sf * (lon_asym**1.06), 0.0, 1.0)
    if disaster_peak_count > 0 and longitude_asymmetry_strength > 0.2:
        g = scatter_disaster_lon_peaks(
            g,
            rng,
            n_peaks=int(disaster_peak_count),
            width=width,
            height=height,
            strength=float(longitude_asymmetry_strength) * 0.55,
            periodic_x=periodic_x,
        )
    gc = np.clip(g * c, 0.0, 1.0)
    overpop_thr = float(np.percentile(gc, 92.0))
    overpop = np.clip((gc - overpop_thr) / max(1.0 - overpop_thr, 1e-8), 0.0, 1.0)
    g = np.clip(g * (1.0 + 0.6 * overpop), 0.0, 1.0)

    base_band = build_equirect_density_map(
        width,
        height,
        disk_height=disk_height,
        halo_fraction=halo_fraction,
        halo_power=halo_power,
        band_lat_sigma=band_lat_sigma,
        band_rotation_deg=band_rotation_deg,
        band_curvature_amp=band_curvature_amp,
        cluster_map=None,
    )
    disk_w = np.clip(base_band / (float(np.max(base_band)) + 1e-8), 0.0, 1.0)
    from starsky_gen.structure_envelope import apply_variable_band_thickness, soften_band_envelope

    disk_w, curve_kind, disk_thickness_mod = apply_variable_band_thickness(
        disk_w,
        rng,
        band_lat_sigma=band_lat_sigma,
        jitter_strength=0.92,
        band_curvature_amp=band_curvature_amp,
        thickness_asymmetry=band_thickness_asymmetry,
        mesoscale_strength=disk_mesoscale_thickness_strength,
        periodic_x=periodic_x,
    )
    disk_w = soften_band_envelope(
        disk_w, (height, width), periodic_x=periodic_x, lat_blur_sigma=10.0, power=0.54
    )
    disk_w = np.clip(disk_w * (0.80 + 0.20 * lon_asym), 0.0, 1.0)
    from starsky_gen.structure_envelope import (
        apply_disk_weight_pole_falloff,
        soften_disk_weight_band_rim,
    )

    disk_w = soften_disk_weight_band_rim(disk_w, periodic_x=periodic_x, strength=0.48)
    disk_w = apply_disk_weight_pole_falloff(disk_w, height, sigma=0.36, power=1.24)

    erosion = build_filament_erosion_map(rng, height, width, periodic_x=periodic_x)
    erosion = np.clip(np.maximum(erosion, scar * 0.72), 0.0, 1.0)
    fractal_dark = build_fractal_from_latent_ridge(latent_ridge, height, width)
    fractal_dark = np.clip(np.maximum(fractal_dark, erosion * 0.45), 0.0, 1.0)
    disruption = build_band_disruption_field(rng, height, width, periodic_x=periodic_x)
    ext_maps = MorphologyExtinctionMaps(
        erosion=erosion.astype(np.float64),
        fractal_dark=fractal_dark.astype(np.float64),
        disruption=disruption.astype(np.float64),
    )

    cut_s = float(np.clip(discontinuity_cut_strength, 0.0, 1.2))
    if cut_s > 1e-6:
        disc = ext_maps.disruption
        g = np.clip(g * np.clip(1.0 - disc**1.8 * cut_s, 0.08, 1.0), 0.0, 1.0)

    u_prior = build_unresolved_intensity_from_master(
        g, d, sf, disk_w, void_mask, latent_turb=latent_turb, dust_absorption=dust_a
    )
    sh_c, sw_c = max(4, height // 8), max(8, width // 8)
    sh_m, sw_m = max(8, height // 4), max(16, width // 4)
    u_coarse = _downsample_field(u_prior, sh_c, sw_c)
    u_coarse = _resize_bilinear(u_coarse, height, width, periodic_x=periodic_x)
    mid_src = np.clip(g * c, 0.0, 1.0)
    u_mid = _downsample_field(mid_src, sh_m, sw_m)
    u_mid = _resize_bilinear(u_mid, height, width, periodic_x=periodic_x)
    u_mid = np.clip(u_mid * 0.028, 0.0, 0.08)

    structure_survival, vertical_extent, lat_shift = build_vertical_structure_envelope(
        height,
        width,
        rng,
        g,
        disk_w,
        band_lat_sigma=band_lat_sigma,
        extent_strength=vertical_extent_strength,
        host_latitude_scale=structure_host_latitude_scale,
        periodic_x=periodic_x,
        mesoscale_field=disk_thickness_mod,
    )
    from starsky_gen.structure_envelope import reinject_vertical_dust_structure

    dust_a = reinject_vertical_dust_structure(
        dust_a,
        vertical_extent,
        structure_survival,
        latent_turb,
        latent_ridge_hi,
        disk_w,
        periodic_x=periodic_x,
        strength=float(np.clip(vertical_extent_strength * 1.08, 0.52, 1.22)),
    )
    from starsky_gen.dust_field import remap_band_absorption_contrast

    dust_a = remap_band_absorption_contrast(
        dust_a, disk_w, floor=0.08, span=0.72, p_lo=8.0, p_hi=86.0
    )
    obliteration = build_obliteration_mask(
        g, dust_a, void_mask, disk_w, strength=obliteration_strength
    )
    brutal = build_brutal_erasure_mask(
        dust_a,
        void_mask,
        obliteration,
        disk_w,
        erosion,
        strength=brutal_erasure_strength,
    )
    gold_pop = build_gold_population_field(
        sf,
        g,
        latent_turb,
        latent_ridge,
        disk_w,
        structure_survival,
        patchiness=gold_population_patchiness,
    )
    from starsky_gen.dust_field import _band_host_gate

    bg_fil = _band_host_gate(disk_w, (height, width))
    lane_soft = gaussian_blur_pil(
        np.clip(erosion * 0.50 + fractal_dark * 0.28, 0.0, 1.0),
        float(np.clip(max(height, width) * 0.012, 1.2, 10.0)),
        periodic_x=periodic_x,
    )
    lane_hp = np.clip(
        np.clip(erosion * 0.50 + fractal_dark * 0.28, 0.0, 1.0) - lane_soft * 0.82,
        0.0,
        1.0,
    )
    dust_a = np.clip(dust_a + lane_hp * 0.18 * bg_fil, 0.06, 0.82)
    scale_f = float(max(height, width))
    med_sig_f = float(np.clip(scale_f * 0.018, 2.5, 20.0))
    a_med_f = gaussian_blur_pil(dust_a, med_sig_f, periodic_x=periodic_x)
    a_hp_f = np.clip(dust_a - a_med_f * 0.76, -0.28, 0.28)
    dust_a = np.clip(dust_a + a_hp_f * 0.28 * bg_fil, 0.06, 0.80)
    from starsky_gen.dust_field import attenuate_column_comb

    if max(height, width) >= 512:
        from starsky_gen.dust_field import attenuate_column_comb

        dust_a = attenuate_column_comb(
            dust_a, disk_w, strength=0.10, periodic_x=periodic_x
        )
        from starsky_gen.structure_envelope import (
            build_fine_puff_field,
            build_structure_morph_host,
            decouple_dust_from_band_gate,
            inject_band_cloud_puffs,
        )

        puff_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) ^ 0xC10D05)
        puff_field = build_fine_puff_field(
            puff_rng, height, width, periodic_x=periodic_x, strength=1.0, center_boost=1.0
        )
        morph_host = build_structure_morph_host(
            disk_w,
            vertical_extent,
            puff_field,
            periodic_x=periodic_x,
        )
        dust_a = decouple_dust_from_band_gate(
            dust_a, morph_host, disk_w, periodic_x=periodic_x, strength=0.78
        )
        if micro_s > 1e-6:
            dust_a = inject_band_cloud_puffs(
                dust_a,
                disk_w,
                puff_rng,
                periodic_x=periodic_x,
                strength=0.58 * micro_s,
            )
            dust_a = decouple_dust_from_band_gate(
                dust_a, morph_host, disk_w, periodic_x=periodic_x, strength=0.62
            )
    d = transmission_from_absorption_map(dust_a, void_floor=0.06, sharpness=2.55)
    if micro_s > 1e-6:
        d = carve_dust_transmission_microstructure(
            d,
            dust_a,
            periodic_x=periodic_x,
            strength=micro_s,
            void_floor=0.06,
            band_weight=disk_w,
        )
    resolve_w = compute_resolve_weight_field(
        g,
        d,
        disk_w,
        drop_strength=drop_strength,
        obliteration=obliteration,
        brutal_erasure=brutal,
    )
    resolve_w = inherit_resolve_weight_from_unresolved(resolve_w, u_prior)
    resolve_w = local_field_variance_stretch(
        resolve_w, variance=lv * 0.85, floor=0.02, ceiling=1.0
    )
    v = np.clip(0.50 + 0.26 * latent_turb + 0.16 * disk_w + 0.12 * d, 0.35, 1.35)

    return GalacticMorphology(
        width=width,
        height=height,
        latent_ridge=latent_ridge.astype(np.float64),
        latent_turb=latent_turb.astype(np.float64),
        stellar_density=g.astype(np.float64),
        star_formation=sf.astype(np.float64),
        dust_absorption=dust_a.astype(np.float64).copy(),
        dust_transmission=d.astype(np.float64).copy(),
        dust_absorption_morph=dust_a.astype(np.float64),
        dust_transmission_morph=d.astype(np.float64),
        cluster_prob=c,
        unresolved_prior=u_prior,
        unresolved_coarse=u_coarse.astype(np.float64),
        unresolved_mid=u_mid.astype(np.float64),
        resolve_weight=resolve_w.astype(np.float64),
        void_mask=void_mask.astype(np.float64),
        obliteration_mask=obliteration.astype(np.float64),
        brutal_erasure_mask=brutal.astype(np.float64),
        structure_survival=structure_survival.astype(np.float64),
        vertical_extent=vertical_extent.astype(np.float64),
        lat_structure_shift=lat_shift.astype(np.float64),
        lon_asymmetry=lon_asym.astype(np.float64),
        gold_population_weight=gold_pop.astype(np.float64),
        psf_environment=v.astype(np.float64),
        disk_weight=disk_w.astype(np.float64),
        disk_thickness_modulation=disk_thickness_mod.astype(np.float64),
        unresolved_accum=np.zeros((height, width), dtype=np.float64),
        extinction_maps=ext_maps,
    )


build_galactic_structure = build_galactic_morphology


def build_population_placement_maps(
    structure: GalacticMorphology,
    *,
    gradient_strength: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    """Scattered-young vs clustered-old placement weights (same shape as density G)."""
    s = float(np.clip(gradient_strength, 0.0, 1.2))
    g = np.clip(structure.stellar_density, 0.0, 1.0)
    c = np.clip(structure.cluster_prob, 0.0, 1.0)
    gp = np.clip(structure.gold_population_weight, 0.0, 1.0)
    sf = np.clip(structure.star_formation, 0.0, 1.0)
    young = np.clip((1.0 - gp) ** 1.02 * (0.30 + 0.70 * sf), 0.0, 1.0)
    old = np.clip(gp**1.06 * (0.40 + 0.60 * (1.0 - sf * 0.82)), 0.0, 1.0)
    scattered = np.clip(g * (0.55 + s * 0.50 * young - s * 0.32 * old), 1e-4, 1.0)
    clustered = np.clip(g * c * (0.20 + s * 1.22 * old) + g * young * sf * 0.06, 0.0, 1.0)
    scattered /= float(np.max(scattered)) + 1e-8
    clustered /= float(np.max(clustered)) + 1e-8
    return scattered.astype(np.float64), clustered.astype(np.float64)


def sample_hierarchical_from_structure(
    rng: np.random.Generator,
    n: int,
    structure: GalacticMorphology,
    *,
    poisson_min_sep_bright_px: float = 14.0,
    poisson_min_sep_faint_px: float = 3.0,
    bright_fraction: float = 0.028,
    association_fraction: float = 0.14,
    population_gradient_strength: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = structure.width, structure.height
    master = structure.stellar_density
    grad = float(np.clip(population_gradient_strength, 0.0, 1.2))
    scattered_map, cluster_peak_map = build_population_placement_maps(
        structure, gradient_strength=grad
    )
    assoc_frac = float(
        np.clip(association_fraction + 0.11 * grad, 0.10, 0.28)
    )
    n_base = max(1, int(n * (1.0 - assoc_frac)))
    n_bright_tier = min(max(1, int(n_base * bright_fraction)), 160)
    faint_sep = float(poisson_min_sep_faint_px) * (1.0 + 0.38 * grad)
    lon_b, lat_b = sample_lon_lat_poisson(
        n_base,
        width,
        height,
        rng,
        density_map=scattered_map if grad > 1e-5 else master,
        min_sep_bright_px=poisson_min_sep_bright_px,
        min_sep_faint_px=faint_sep,
        n_bright=n_bright_tier,
        cluster_strength=0.0,
    )
    n_assoc = max(0, n - n_base)
    if n_assoc < 8:
        from starsky_gen.placement import _normalize_placement_count

        return _normalize_placement_count(lon_b, lat_b, n, master, width, height, rng)
    n_sites = int(np.clip(n_assoc / 16, 8, 48))
    peak_map = (
        cluster_peak_map
        if grad > 1e-5
        else structure.stellar_density * structure.cluster_prob
    )
    gp = structure.gold_population_weight
    rows, cols = pick_association_peaks(peak_map, rng, n_peaks=n_sites, min_sep_px=26.0)
    from starsky_gen.placement import _normalize_placement_count, pixels_to_lon_lat

    lon_parts = [lon_b]
    lat_parts = [lat_b]
    per_site = max(6, n_assoc // max(len(rows), 1))
    for r, c in zip(rows, cols, strict=False):
        n_here = int(rng.integers(max(4, per_site - 4), per_site + 6))
        ri, ci = int(np.clip(int(r), 0, height - 1)), int(c) % width
        gp_site = float(gp[ri, ci])
        n_boost = 1.0 + grad * 0.55 * gp_site
        n_here = max(6, min(int(n_here * n_boost), per_site + 14))
        lon_c, lat_c = pixels_to_lon_lat(
            np.full(n_here, r, dtype=np.float64),
            np.full(n_here, c, dtype=np.float64),
            width,
            height,
            rng,
        )
        tight = 0.50 + 0.50 * (1.0 - gp_site * grad)
        sig_lon = float(rng.uniform(0.006, 0.032)) * tight
        sig_lat = float(rng.uniform(0.005, 0.024)) * tight
        lon_c = (lon_c + rng.normal(0.0, sig_lon, size=n_here)) % (2.0 * np.pi)
        from starsky_gen.placement import lat_equirect_clip_bounds

        lat_lo, lat_hi = lat_equirect_clip_bounds()
        lat_c = np.clip(lat_c + rng.normal(0.0, sig_lat, size=n_here), lat_lo, lat_hi)
        lon_parts.append(lon_c)
        lat_parts.append(lat_c)
    lon = np.concatenate(lon_parts)
    lat = np.concatenate(lat_parts)
    return _normalize_placement_count(lon, lat, n, master, width, height, rng)


sample_hierarchical_from_morphology = sample_hierarchical_from_structure


def sample_foreground_lon_lat_from_structure(
    rng: np.random.Generator,
    n: int,
    structure: GalacticMorphology,
) -> tuple[np.ndarray, np.ndarray]:
    g = structure.stellar_density
    dw = structure.disk_weight
    fg_map = g * (0.42 + 0.58 * (1.0 - dw) ** 0.85)
    fg_map = fg_map + 0.12 * g * structure.cluster_prob
    fg_map = fg_map / (float(np.max(fg_map)) + 1e-8)
    return sample_lon_lat_poisson(
        n,
        structure.width,
        structure.height,
        rng,
        density_map=np.clip(fg_map, 1e-4, 1.0),
        min_sep_bright_px=10.0,
        min_sep_faint_px=4.0,
        n_bright=max(1, n // 12),
        cluster_strength=0.0,
    )


def attach_dust_visibility_to_catalog(
    catalog: dict[str, np.ndarray],
    structure: GalacticMorphology,
    width: int,
    height: int,
) -> None:
    from starsky_gen.projections import sph_to_equirect_xy

    lon = catalog["lon"]
    lat = catalog["lat"]
    xi, yi = sph_to_equirect_xy(lon, lat, width, height)
    yi = np.clip(yi, 0, structure.height - 1)
    xi = xi % structure.width
    vis = structure.dust_transmission[yi, xi]
    oblit = structure.obliteration_mask[yi, xi]
    brutal = structure.brutal_erasure_mask[yi, xi]
    vis = vis * np.clip(1.0 - oblit * 0.88, 0.04, 1.0)
    vis = vis * np.clip(1.0 - brutal * 0.96, 0.04, 1.0)
    catalog["dust_visibility"] = np.clip(vis**0.84, 0.04, 1.0).astype(np.float64)


def attach_stellar_population_to_catalog(
    catalog: dict[str, np.ndarray],
    structure: GalacticMorphology,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Per-star gold-population weight (0=young/no gold bias, 1=old/gold bias)."""
    from starsky_gen.color_science import (
        adjust_bv_for_population,
        adjust_teffective_for_population,
    )
    from starsky_gen.projections import sph_to_equirect_xy

    lon = catalog["lon"]
    lat = catalog["lat"]
    xi, yi = sph_to_equirect_xy(lon, lat, width, height)
    yi = np.clip(yi, 0, structure.height - 1)
    xi = xi % structure.width
    gp = structure.gold_population_weight[yi, xi]
    catalog["gold_population"] = np.clip(gp, 0.0, 1.0).astype(np.float64)
    if "teff" in catalog:
        catalog["teff"] = adjust_teffective_for_population(
            catalog["teff"], catalog["gold_population"], rng
        )
    if "bv" in catalog:
        bv = adjust_bv_for_population(
            catalog["bv"], catalog["gold_population"], rng
        )
        catalog["bv"] = bv
        catalog["color_idx"] = np.select(
            [
                bv < 0.12,
                (bv >= 0.12) & (bv < 0.42),
                (bv >= 0.42) & (bv < 0.95),
            ],
            [1, 0, 2],
            default=3,
        ).astype(np.int64)


def compute_resolve_weight_field(
    g: np.ndarray,
    d: np.ndarray,
    disk_w: np.ndarray,
    *,
    drop_strength: float,
    obliteration: np.ndarray | None = None,
    brutal_erasure: np.ndarray | None = None,
) -> np.ndarray:
    """Plane resolve weight W from stellar density, dust transmission, and disk."""
    drop = float(np.clip(drop_strength, 0.0, 1.0))
    w = np.clip(
        (0.38 + 0.62 * g)
        * (0.18 + 0.82 * d)
        * (1.0 - drop * (disk_w**1.6)),
        0.02,
        1.0,
    )
    if obliteration is not None:
        obl = np.clip(np.asarray(obliteration, dtype=np.float64), 0.0, 1.0)
        w = w * np.clip(1.0 - obl**2.35, 0.02, 1.0)
    if brutal_erasure is not None:
        br = np.clip(np.asarray(brutal_erasure, dtype=np.float64), 0.0, 1.0)
        w = w * np.clip(1.0 - br**2.85, 0.02, 1.0)
    return w.astype(np.float64)


def build_unresolved_intensity_from_master(
    g: np.ndarray,
    d: np.ndarray,
    sf: np.ndarray,
    disk_w: np.ndarray,
    void_mask: np.ndarray,
    *,
    latent_turb: np.ndarray | None = None,
    dust_absorption: np.ndarray | None = None,
) -> np.ndarray:
    """Inherited unresolved luminosity field: master_density → unresolved intensity.

    Coupling note: raw u∝g×T damps the whole band when extinction T is low (morph-primary),
    which reads as a smooth gray wash after merge. d_open softens that; turb/dust-open HF keeps
    lane/puff contrast for the speckle pyramid.
    """
    dw = np.clip(np.asarray(disk_w, dtype=np.float64), 0.0, 1.0)
    d_open = np.clip(np.asarray(d, dtype=np.float64), 0.0, 1.0) ** 0.82
    u_base = np.clip(
        g * (0.38 + 0.62 * d_open) * sf * (0.50 + 0.50 * dw) * (1.0 - void_mask * 0.85),
        0.0,
        None,
    )
    if latent_turb is not None and dust_absorption is not None:
        from starsky_gen.procedural_noise import gaussian_blur_pil

        turb = np.clip(np.asarray(latent_turb, dtype=np.float64), 0.0, 1.0)
        dust = np.clip(np.asarray(dust_absorption, dtype=np.float64), 0.0, 1.0)
        sig = float(np.clip(max(g.shape) * 0.007, 0.45, 8.0))
        turb_hp = np.clip(turb - gaussian_blur_pil(turb, sig, periodic_x=True) * 0.74, 0.0, 1.0)
        dust_open = np.clip(1.0 - dust, 0.0, 1.0) ** 1.05
        u_base = np.clip(
            u_base * (0.72 + 0.28 * dust_open * dw) + turb_hp * dw * 0.014, 0.0, None
        )
    u_base = u_base / (float(np.percentile(u_base, 99.5)) + 1e-8)
    return np.clip(u_base * 0.042, 0.0, 0.12).astype(np.float64)


def inherit_resolve_weight_from_unresolved(
    resolve_w: np.ndarray,
    unresolved_u: np.ndarray,
    *,
    coupling: float = 0.88,
) -> np.ndarray:
    """Resolved probability falls where unresolved intensity is high (shared inheritance)."""
    u = np.asarray(unresolved_u, dtype=np.float64)
    u = u / (float(np.percentile(u, 99.2)) + 1e-8)
    u = np.clip(u, 0.0, 1.0)
    c = float(np.clip(coupling, 0.0, 1.0))
    return np.clip(np.asarray(resolve_w, dtype=np.float64) * (1.0 - c * u), 0.04, 1.0).astype(
        np.float64
    )


_build_unresolved_prior = build_unresolved_intensity_from_master


def _apply_macro_voids(
    g: np.ndarray,
    rng: np.random.Generator,
    *,
    count: int,
    height: int,
    width: int,
    strength: float,
    periodic_x: bool = True,
) -> np.ndarray:
    """Rare torn voids along the plane (ridged tears, not symmetric ellipses)."""
    from starsky_gen.dust_field import _blur_y_only_field
    from starsky_gen.procedural_noise import _resize_bilinear, ridged_fbm2d

    s = float(np.clip(strength, 0.0, 1.0))
    if s < 1e-6:
        return g
    h, w = int(height), int(width)
    ch, cw = max(6, h // 28), max(10, w // 16)
    tear = ridged_fbm2d(rng, ch, cw, base_scale=0.10, octaves=5, periodic_x=periodic_x)
    tear = _resize_bilinear(tear, h, w, periodic_x=periodic_x)
    tear = _blur_y_only_field(tear, 2.4)
    tear = np.clip((tear - 0.68) / 0.24, 0.0, 1.0) ** 1.35
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    plane = np.exp(-((yy**2) / 0.42))
    n = max(1, int(count))
    out = g.copy()
    for i in range(n):
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + (i + 1) * 9973)
        shard = ridged_fbm2d(
            sub, max(4, ch // 2), max(6, cw // 2), base_scale=0.14 + 0.04 * i, octaves=4, periodic_x=periodic_x
        )
        shard = _resize_bilinear(shard, h, w, periodic_x=periodic_x)
        shard = np.clip((shard - float(sub.uniform(0.62, 0.78))) / 0.22, 0.0, 1.0)
        carve = np.clip(tear * shard * plane * float(sub.uniform(0.45, 0.95)) * s, 0.0, 0.72)
        out *= 1.0 - carve
    return np.clip(out, 0.0, 1.0)


def deposit_catalog_unresolved_flux(
    catalog: dict[str, np.ndarray],
    morphology: GalacticMorphology,
    rng: np.random.Generator,
    *,
    width: int,
    height: int,
    magnitude_ref_mag: float,
    mag_bright: float,
    mag_faint: float,
    dropout_strength: float,
    mid_layer: bool = False,
    layer_flux_scale: float = 1.0,
) -> np.ndarray:
    """Deposit withheld flux for stars that fail resolution; returns skip mask for paint."""
    from starsky_gen.projections import sph_to_equirect_xy
    from starsky_gen.psf import flux_from_mag

    n = int(catalog["lon"].size)
    skip = np.zeros(n, dtype=bool)
    if n < 1 or "phot_mag" not in catalog:
        return skip
    xi, yi = sph_to_equirect_xy(catalog["lon"], catalog["lat"], width, height)
    yi = np.clip(yi, 0, morphology.height - 1)
    xi = xi % morphology.width
    dust_vis = catalog.get("dust_visibility")
    mags = catalog["phot_mag"].astype(np.float64)
    scale = float(layer_flux_scale)
    for i in range(n):
        yi_px, xi_px = int(yi[i]), int(xi[i])
        rw = morphology.sample_resolve_weight(yi_px, xi_px)
        p_keep = resolved_keep_probability(
            g=morphology.sample_stellar(yi_px, xi_px),
            dust_t=morphology.sample_dust(yi_px, xi_px),
            disk_w=float(morphology.disk_weight[yi_px, xi_px]),
            mag=float(mags[i]),
            mag_bright=mag_bright,
            mag_faint=mag_faint,
            dropout_strength=dropout_strength,
            mid_layer=mid_layer,
            resolve_weight=rw,
            obliteration=morphology.sample_obliteration(yi_px, xi_px),
            brutal_erasure=morphology.sample_brutal_erasure(yi_px, xi_px),
        )
        if rng.random() >= p_keep:
            flux = float(flux_from_mag(float(mags[i]), magnitude_ref_mag)) * scale
            if dust_vis is not None:
                flux *= float(dust_vis[i])
            morphology.deposit_unresolved(yi_px, xi_px, flux)
            skip[i] = True
    catalog["_resolved_skip"] = skip
    return skip


def resolved_keep_probability(
    *,
    g: float,
    dust_t: float,
    disk_w: float,
    mag: float | None,
    mag_bright: float,
    mag_faint: float,
    dropout_strength: float,
    mid_layer: bool,
    resolve_weight: float | None = None,
    obliteration: float | None = None,
    brutal_erasure: float | None = None,
) -> float:
    """P(star is resolved); uses explicit resolve_weight field when provided."""
    if resolve_weight is not None:
        base = float(np.clip(resolve_weight, 0.02, 1.0))
    else:
        drop = float(np.clip(dropout_strength, 0.0, 1.0))
        if mid_layer:
            drop *= 0.58
        dw = float(np.clip(disk_w, 0.0, 1.0))
        base = 1.0 - drop * (dw**1.45)
        base *= 0.32 + 0.68 * float(np.clip(g, 0.0, 1.0))
        base *= 0.28 + 0.72 * float(np.clip(dust_t, 0.0, 1.0))
    if mag is not None:
        span = max(float(mag_faint) - float(mag_bright), 1e-6)
        faint_u = float(np.clip((float(mag) - float(mag_bright)) / span, 0.0, 1.0))
        base *= 1.0 - 0.78 * (faint_u**1.85)
        if float(mag) < float(mag_bright) + 0.5:
            base = max(base, 0.94)
    if obliteration is not None and float(obliteration) > 0.06:
        obl = float(np.clip(obliteration, 0.0, 1.0))
        base *= (1.0 - min(1.0, obl * 1.75)) ** 2.65
        base = max(base, 0.012)
    if brutal_erasure is not None and float(brutal_erasure) > 0.08:
        br = float(np.clip(brutal_erasure, 0.0, 1.0))
        base *= (1.0 - min(1.0, br * 1.92)) ** 3.15
        base = max(base, 0.05)
    return float(np.clip(base, 0.05, 1.0))


def _add_luminous_continuum(
    canvas: np.ndarray,
    intensity: np.ndarray,
    *,
    texture_strength: float,
    periodic_x: bool,
    blur_sigma: float = 1.55,
    amp: float = 0.085,
    continuum_rgb: np.ndarray | None = None,
    morph_hf: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth additive HDR base from inherited unresolved intensity (before speckle).

    Intentionally low-pass: unresolved_prior is already g×d weighted. morph_hf re-injects turbulent
    pockets so the pre-extinction canvas is not a single Gaussian band wash.
    """
    from starsky_gen.procedural_noise import gaussian_blur_pil

    t = float(np.clip(texture_strength, 0.0, 2.0))
    if t < 1e-6:
        return canvas
    from starsky_gen.structure_envelope import diffuse_scale_hierarchy

    raw = np.asarray(intensity, dtype=np.float64)
    cont = gaussian_blur_pil(raw, blur_sigma, periodic_x=periodic_x)
    cont = cont / (float(np.percentile(cont, 99.0)) + 1e-8)
    cont = diffuse_scale_hierarchy(cont, strength=0.22, periodic_x=periodic_x)
    cont_hi = np.clip(raw - gaussian_blur_pil(raw, blur_sigma * 0.55, periodic_x=periodic_x) * 0.72, 0.0, 1.0)
    lift = np.clip(cont * t * amp + cont_hi * t * amp * 0.55, 0.0, 0.30)
    if morph_hf is not None:
        hf = np.clip(np.asarray(morph_hf, dtype=np.float64), 0.0, 1.0)
        lift = np.clip(lift + hf * t * amp * 0.68, 0.0, 0.36)
    if continuum_rgb is None:
        rgb = np.array([0.988, 0.992, 0.998], dtype=np.float64)
    else:
        rgb = np.asarray(continuum_rgb, dtype=np.float64).reshape(3)
    return np.maximum(0.0, canvas + lift[..., None] * rgb)


def _morph_turbulent_hf_field(structure: GalacticMorphology, *, periodic_x: bool = True) -> np.ndarray:
    """HF guide from morph dust/turb (pockets between lanes, not blurred wash)."""
    from starsky_gen.procedural_noise import gaussian_blur_pil

    dust = np.clip(structure.dust_absorption_morph, 0.0, 1.0)
    turb = np.clip(structure.latent_turb, 0.0, 1.0)
    sig = float(np.clip(max(dust.shape) * 0.006, 0.35, 6.0))
    dust_hp = np.clip(dust - gaussian_blur_pil(dust, sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0)
    turb_hp = np.clip(turb - gaussian_blur_pil(turb, sig, periodic_x=periodic_x) * 0.74, 0.0, 1.0)
    open_ = np.clip(1.0 - dust, 0.0, 1.0) ** 1.06
    return np.clip(open_ * 0.42 + turb_hp * 0.38 + dust_hp * 0.20, 0.0, 1.0).astype(np.float64)


def compose_inherited_unresolved_field(
    canvas: np.ndarray,
    rng: np.random.Generator,
    structure: GalacticMorphology,
    *,
    texture_strength: float = 1.0,
    periodic_x: bool = True,
    deposit_primary: bool = True,
    unresolved_rgb: np.ndarray | None = None,
    preserve_morph_hf: bool = True,
) -> np.ndarray:
    """Paint unresolved luminous field on canvas before extinction (G → U → speckle).

    Main morphology-smoothing stage in the star+ISM stack: continuum blur then Poisson speckle on
    blurred rate maps. preserve_morph_hf adds turbulent guide on tier-0 and a fine morph speckle pass.
    """
    morph_hf = _morph_turbulent_hf_field(structure, periodic_x=periodic_x) if preserve_morph_hf else None
    out = _add_luminous_continuum(
        canvas,
        structure.unresolved_prior,
        texture_strength=texture_strength,
        periodic_x=periodic_x,
        blur_sigma=1.12,
        amp=0.072,
        continuum_rgb=unresolved_rgb,
        morph_hf=morph_hf,
    )
    return apply_unresolved_field_to_canvas(
        out,
        rng,
        structure,
        texture_strength=texture_strength,
        periodic_x=periodic_x,
        deposit_primary=deposit_primary,
        speckle_rgb=unresolved_rgb,
    )


def _poisson_speckle_layer(
    canvas: np.ndarray,
    rng: np.random.Generator,
    rate_map: np.ndarray,
    *,
    amp_scale: float,
    p_scale: float,
    speckle_rgb: np.ndarray | None = None,
) -> np.ndarray:
    h, w = rate_map.shape
    u = rate_map / (float(np.percentile(rate_map, 99.0)) + 1e-8)
    u = np.clip(u * amp_scale, 0.0, 0.12)
    p = np.clip(u * p_scale, 0.0, 0.09)
    amp = np.clip(u * (0.80 + 0.40 * rng.random((h, w))), 0.0, 0.07)
    speckle = np.where(rng.random((h, w)) < p, amp, 0.0)
    if speckle_rgb is None:
        rgb = np.array([0.992, 0.994, 0.996], dtype=np.float64)
    else:
        rgb = np.asarray(speckle_rgb, dtype=np.float64).reshape(3)
    return np.maximum(0.0, canvas + speckle[..., None] * rgb)


def apply_unresolved_field_to_canvas(
    canvas: np.ndarray,
    rng: np.random.Generator,
    structure: GalacticMorphology,
    *,
    texture_strength: float = 1.0,
    periodic_x: bool = True,
    deposit_primary: bool = True,
    speckle_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Poisson speckle from deposit pyramid; weak u_prior only in low-deposit gaps."""
    from starsky_gen.placement import equirect_steradian_weights
    from starsky_gen.procedural_noise import gaussian_blur_pil

    t = float(np.clip(texture_strength, 0.0, 2.0))
    if t < 1e-6:
        return canvas
    out = canvas
    h, w = structure.height, structure.width
    area_w = equirect_steradian_weights(h, w)
    inherit_rate = structure.unresolved_speckle_rate() * area_w
    parent_seed = int(rng.integers(0, 2**31))

    from starsky_gen.structure_envelope import diffuse_scale_hierarchy, latitude_plane_gate

    plane_gate = latitude_plane_gate(h, sigma=0.42, power=1.10)
    morph_speckle_gate = latitude_plane_gate(h, sigma=0.58, power=0.94)

    if deposit_primary:
        accum = structure.unresolved_accum.astype(np.float64)
        acc_norm = accum / (float(np.percentile(accum, 99.2)) + 1e-8)
        gaps = np.clip(1.0 - acc_norm, 0.0, 1.0)
        # σ-pyramid: each tier blurs the deposit field before Poisson speckle — cap σ so we do not
        # smear puff/lane HF that dust_A already has (legacy went to σ=6.5).
        scales = (
            (0.22, 1.12, 5.2),
            (0.55, 1.02, 3.6),
            (1.35, 0.92, 2.0),
            (2.8, 0.48, 0.72),
        )
        morph_hf = _morph_turbulent_hf_field(structure, periodic_x=periodic_x)
        for i, (sigma, amp_scale, p_scale) in enumerate(scales):
            layer_rng = np.random.default_rng(parent_seed + i * 7919)
            src = (accum + structure.unresolved_prior * 0.35) * plane_gate
            if i == 0:
                src = np.clip(src + morph_hf * inherit_rate * 0.28, 0.0, None)
            blurred = gaussian_blur_pil(src, sigma, periodic_x=periodic_x)
            rate = blurred * inherit_rate * t
            out = _poisson_speckle_layer(
                out, layer_rng, rate, amp_scale=amp_scale, p_scale=p_scale, speckle_rgb=speckle_rgb
            )
        fine_rng = np.random.default_rng(parent_seed + 0xF1AE)
        # Fine morph speckle: unblurred rate map so tight clouds survive the pyramid.
        fine_rate = morph_hf * morph_speckle_gate * inherit_rate * t * 2.05
        out = _poisson_speckle_layer(
            out, fine_rng, fine_rate, amp_scale=1.12, p_scale=5.8, speckle_rgb=speckle_rgb
        )
        gap_prior = structure.unresolved_prior * gaps * plane_gate * area_w * t * 0.06
        out = _poisson_speckle_layer(
            out, rng, gap_prior, amp_scale=0.32, p_scale=0.80, speckle_rgb=speckle_rgb
        )
    else:
        u_fine = structure.unresolved_total()
        out = _poisson_speckle_layer(
            out, rng, u_fine * t, amp_scale=1.0, p_scale=3.2, speckle_rgb=speckle_rgb
        )
        rng_mid = np.random.default_rng(parent_seed + 1)
        out = _poisson_speckle_layer(
            out,
            rng_mid,
            structure.unresolved_mid * t,
            amp_scale=0.85,
            p_scale=1.4,
            speckle_rgb=speckle_rgb,
        )
        rng_coarse = np.random.default_rng(parent_seed + 2)
        out = _poisson_speckle_layer(
            out,
            rng_coarse,
            structure.unresolved_coarse * t,
            amp_scale=0.55,
            p_scale=0.65,
            speckle_rgb=speckle_rgb,
        )
    return np.clip(out, 0.0, None)


def morphology_grayscale_preview(morphology: GalacticMorphology) -> np.ndarray:
    """Diagnostic: G + (1-D) + U without grading."""
    g = morphology.stellar_density
    d = morphology.dust_transmission
    u = morphology.unresolved_total()
    u = u / (float(np.max(u)) + 1e-8)
    return np.clip(0.4 * g + 0.35 * (1.0 - d) + 0.25 * u, 0.0, 1.0).astype(np.float64)
