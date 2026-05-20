from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, field_validator

NebulaDebugPass: TypeAlias = Literal[
    "normal",
    "occluder_only",
    "continuum_only",
    "layer_base",
    "layer_filaments",
    "layer_fine",
    "layer_carve",
    "layer_dust_alpha",
    "mask_only",
    "warp_vectors",
]


class NebulaMode(str, Enum):
    distant = "distant"
    full = "full"
    galaxy_streak = "galaxy_streak"


class ProjectionMode(str, Enum):
    equirectangular = "equirectangular"
    cubemap = "cubemap"
    both = "both"


class OutputFormat(str, Enum):
    png = "png"
    jpg = "jpg"


class RenderProfile(str, Enum):
  physical = "physical"
  physical_grade = "physical_grade"
  full = "full"


class FeatureConfig(BaseModel):
    stars: bool = True
    depth: bool = True
    nebula: bool = True
    galaxy_view: bool = True
    reference_anchors: bool = Field(
        True,
        description="Paint a small set of bright real stars at approximate Galactic (l,b) positions.",
    )
    background_gradient: bool = True
    black_background: bool = False
    jpeg_artifact_pass: bool = False
    long_exposure_look: bool = Field(
        True,
        description=(
            "Asymmetric stacked-Milky-Way look: tilted sky floor, mild corner glow, off-center vignette "
            "(disable for a cleaner symmetric grade)."
        ),
    )
    long_exposure_star_trails: bool = Field(
        True,
        description="Integrate star PSF over subframes (trails) when long_exposure_look; avoids post-blur.",
    )
    long_exposure_subframes: int = Field(
        7,
        ge=3,
        le=15,
        description="Subframe count for long-exposure star trail integration.",
    )
    long_exposure_trail_step_px: float = Field(
        0.55,
        ge=0.15,
        le=2.5,
        description="Half-span of star trail in pixels across subframes.",
    )
    split_star_display_grade: bool = Field(
        True,
        description=(
            "Grade nebula and stars separately: nebula through ISP/tone/bloom, then composite stars "
            "(preserves blue point sources). Recommended for galaxy_streak."
        ),
    )
    disk_radiance_unify_strength: float = Field(
        0.72,
        ge=0.0,
        le=1.0,
        description=(
            "Unify unresolved/nebula diffuse chroma with resolved stars (reduces cyan-white compositing islands)."
        ),
    )
    unresolved_spectral_warmth: float = Field(
        0.58,
        ge=0.0,
        le=1.0,
        description="Tint unresolved speckle/continuum toward disk star chroma (0=neutral blue-white).",
    )
    ism_lift_chroma_lock: float = Field(
        0.72,
        ge=0.0,
        le=1.0,
        description="ISM dominance lift uses star chroma reference instead of independent warm+cool push.",
    )
    unified_linear_grade: bool = Field(
        False,
        description=(
            "Single linear HDR grade for stars+diffuse together (sets split_star_display_grade off). "
            "Best fix for spectral/shoulder mismatch artifacts."
        ),
    )
    split_star_match_scene_tone: bool = Field(
        True,
        description="When split-grading stars, match asinh stretch to in-plane scene HDR before composite.",
    )
    photon_exposure_unify_strength: float = Field(
        0.72,
        ge=0.0,
        le=1.0,
        description="Shared linear exposure for diffuse canvas + resolved stars (same photon law).",
    )
    morphology_local_variance: float = Field(
        1.32,
        ge=0.5,
        le=2.2,
        description="Mean-preserving contrast on dust/survival maps (variance ↑, not mean dust ↑).",
    )
    morphology_obliteration_strength: float = Field(
        0.82,
        ge=0.0,
        le=1.25,
        description="Dense stellar patches extinguished → few survivors, bright neighbors.",
    )
    morphology_regional_chaos: float = Field(
        0.38,
        ge=0.0,
        le=0.9,
        description="Seed-indexed regional SF/dust modulation (environmental variation).",
    )
    structure_vertical_extent: float = Field(
        0.72,
        ge=0.0,
        le=1.25,
        description="Random vertical envelope: clouds protrude/sink off the plane (not global nebula boost).",
    )
    structure_host_latitude_scale: float = Field(
        1.85,
        ge=1.0,
        le=3.5,
        description="Widen band host latitude so structure survives above/below the plane.",
    )
    longitude_asymmetry_strength: float = Field(
        0.88,
        ge=0.0,
        le=1.35,
        description="Left quiet → center fractured → right disaster (longitude modulation).",
    )
    extinction_brutal_erasure_strength: float = Field(
        0.78,
        ge=0.0,
        le=1.25,
        description="Occasional erasure pockets (~5% star survival), not soft dust.",
    )
    extinction_brutal_survival_floor: float = Field(
        0.05,
        ge=0.02,
        le=0.12,
        description="Transmission floor in brutal erasure lanes.",
    )
    diffuse_scale_hierarchy: float = Field(
        0.96,
        ge=0.0,
        le=1.25,
        description="Mix tiny mottling + medium + huge regional diffuse scales.",
    )
    seam_guard_strength: float = Field(
        0.85,
        ge=0.0,
        le=1.2,
        description="Suppress accidental equirect seam hotspots at lon 0/width.",
    )
    off_band_emission_strength: float = Field(
        2.45,
        ge=0.0,
        le=3.5,
        description="H II / red emission off the galactic band (late composite, chroma preserved).",
    )
    off_band_hii_blob_count: int = Field(
        5,
        ge=0,
        le=8,
        description="Number of regional off-band H II clouds per frame (0=diffuse wash only).",
    )
    scene_red_hii_spot_count: int = Field(
        3,
        ge=0,
        le=6,
        description="Additional compact red H II nebulae (band + off-band, pole-gated).",
    )
    scene_red_hii_strength: float = Field(
        1.85,
        ge=0.0,
        le=3.0,
        description="Brightness scale for scene_red_hii_spot_count nebulae.",
    )
    band_hii_patch_count: int = Field(
        4,
        ge=0,
        le=5,
        description="In-band red H II patches along the galactic disk.",
    )
    band_hii_strength: float = Field(
        2.8,
        ge=0.0,
        le=4.0,
        description="Brightness of in-band H II patches (late composite).",
    )
    off_band_hii_diffuse_weight: float = Field(
        0.0,
        ge=0.0,
        le=0.45,
        description="Diffuse red wash under localized H II clouds (0 = discrete complexes only).",
    )
    morphology_detail_strength: float = Field(
        1.34,
        ge=0.6,
        le=2.2,
        description="Additive small/fine ISM detail on top of large-scale morphology envelope.",
    )
    morphology_white_brightness: float = Field(
        1.02,
        ge=0.7,
        le=1.8,
        description="Brightness scale for in-band white ISM in morphology pass.",
    )
    off_band_emit_scale: float = Field(
        0.62,
        ge=0.0,
        le=2.5,
        description="Fraction of procedural line emission routed off-band (decoupled from plane tint).",
    )
    off_band_late_composite: bool = Field(
        True,
        description="Composite off-band red after plane chroma harmonize (avoids red compression).",
    )
    stellar_gold_population_patchiness: float = Field(
        1.38,
        ge=0.6,
        le=2.0,
        description="Regional contrast for old (gold) vs young (neutral/blue) stellar patches in the band.",
    )
    stellar_age_gradient_strength: float = Field(
        0.76,
        ge=0.0,
        le=1.2,
        description="Old=gold/dense/clustered vs young=blue-white/scattered placement and color split.",
    )
    hierarchical_star_placement: bool = Field(
        True,
        description="Master density field → association peaks → Poisson individuals (less salt).",
    )
    extinction_first_nebula: bool = Field(
        True,
        description="Fractal dust mask deepens lanes; emission/haze gated by extinction.",
    )
    extinction_discontinuity_strength: float = Field(
        1.72,
        ge=0.0,
        le=2.0,
        description="Asymmetric voids/gouges in the band (higher = nastier dark cuts, less smooth strip).",
    )
    extinction_void_floor: float = Field(
        0.001,
        ge=0.0005,
        le=0.12,
        description="Transmission floor in deepest lanes (galaxy can nearly vanish).",
    )
    extinction_opacity_gamma: float = Field(
        1.42,
        ge=1.0,
        le=2.6,
        description="Opacity^γ before extinction (γ>1): preserve weak dust, deepen lanes.",
    )
    extinction_filament_strength: float = Field(
        1.44,
        ge=0.0,
        le=1.5,
        description="Filament erosion absorption vs soft cloud (1=fractured rivers, 0=legacy only).",
    )
    extinction_fine_texture_strength: float = Field(
        0.82,
        ge=0.0,
        le=1.0,
        description="Fine (~1–5 px) dust extinction carved on top of macro lanes.",
    )
    emission_extinction_gate_power: float = Field(
        2.28,
        ge=0.5,
        le=3.0,
        description="Exponent on clear-sightline mask for emission/haze (higher = darker lanes).",
    )
    nebula_haze_strength: float = Field(
        0.62,
        ge=0.0,
        le=2.0,
        description="Scale gold haze / band cloud / disk glow additives.",
    )
    band_nebula_radiance_scale: float = Field(
        1.22,
        ge=0.8,
        le=6.0,
        description="Global multiplier on procedural gas/emission in the galactic plane.",
    )
    band_hdr_peak_target: float = Field(
        0.40,
        ge=0.24,
        le=0.55,
        description="Pre-tone-map band luma percentile cap (lower = less overblown plane).",
    )
    band_plane_luma_cap: float = Field(
        0.48,
        ge=0.26,
        le=0.62,
        description="Hard per-pixel linear luma cap in the galactic plane before tone map.",
    )
    band_display_peak_cap: float = Field(
        0.64,
        ge=0.48,
        le=0.72,
        description="Display-space luma cap in the galactic plane after stars/H II composite.",
    )
    band_star_plane_scale: float = Field(
        0.62,
        ge=0.12,
        le=1.0,
        description="Star layer brightness in the bright plane (lower lets nebula dominate).",
    )
    band_ism_dominance: float = Field(
        0.42,
        ge=0.0,
        le=2.0,
        description=(
            "How strongly diffuse ISM should win over point stars in the plane "
            "(lifts band continuum, dims in-plane stars)."
        ),
    )
    band_dark_patch_strength: float = Field(
        0.48,
        ge=0.0,
        le=1.2,
        description="Carve dark dust patches / voids into the luminous band after ISM lift.",
    )
    galactic_band_color_grade: bool = Field(
        True,
        description=(
            "Final display pass: warm gold/white ISM, brown/black extinction lanes (inner bias), "
            "sparse H II red; corrects pale-blue diffuse gas."
        ),
    )
    galactic_band_color_grade_strength: float = Field(
        0.88,
        ge=0.0,
        le=1.2,
        description="Strength of the final galactic band color grade on diffuse/plane pixels.",
    )
    galactic_band_dust_black_strength: float = Field(
        1.12,
        ge=0.0,
        le=1.5,
        description="Turbulent near-black dust carve in the final band color grade.",
    )
    galactic_band_gas_fluff_strength: float = Field(
        0.78,
        ge=0.0,
        le=1.3,
        description="Puffy cleared pockets and micro contrast in the final band color grade.",
    )
    galactic_band_micro_display_strength: float = Field(
        1.32,
        ge=0.0,
        le=1.5,
        description="Final display sculpt of pixel-scale gas filaments and cloud fluff.",
    )
    galactic_band_separation_strength: float = Field(
        1.12,
        ge=0.0,
        le=1.6,
        description="Local contrast expansion — gaps between clouds, lanes, and cleared pockets.",
    )
    volumetric_scatter_strength: float = Field(
        0.38,
        ge=0.0,
        le=1.5,
        description="Scale forward Mie/HG scatter on emission.",
    )
    optics_before_tone_map: bool = Field(
        True,
        description="Apply lens vignette in linear space before scene tone map.",
    )
    star_psf_bloom_scale: float = Field(
        0.42,
        ge=0.0,
        le=1.5,
        description="Scale tight sensor bloom on bright stars only.",
    )
    stars_after_display_grade: bool = Field(
        False,
        description=(
            "Alias for split_star_display_grade when galaxy_streak (legacy flag). "
            "If both set, split display composite is used."
        ),
    )
    dust_multiscatter_strength: float = Field(
        0.008,
        ge=0.0,
        le=0.15,
        description="Secondary scatter fill into dark dust lanes (lower = less smoky lane fill).",
    )
    nebula_spiral_strength: float = Field(
        0.52,
        ge=0.0,
        le=1.0,
        description="Log-spiral arm density modulator on galaxy nebula noise.",
    )
    nebula_turbulence_octaves: int = Field(
        3,
        ge=1,
        le=3,
        description="FBM+curl octave bands for nebula (1–3 scales).",
    )
    background_texture_strength: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Scale for unresolved-star background texture (0=smooth, 1=default, 2=grittier).",
    )
    photoreal_stars: bool = Field(
        True,
        description=(
            "Magnitude–flux sampling + compact Moffat PSF stamping for galaxy background/cluster stars "
            "(classic stamp path when disabled)."
        ),
    )
    star_psf_variety: bool = Field(
        True,
        description=(
            "Per-star PSF family (pinprick, soft seeing, saturated, extincted, stack twin) "
            "weighted by magnitude, extinction, and crowding."
        ),
    )
    mag_bright_lim: float = Field(
        8.0,
        ge=-2.0,
        le=14.0,
        description="Bright end of procedural apparent-magnitude range (lower m → brighter stars).",
    )
    mag_faint_lim: float = Field(
        20.0,
        ge=8.0,
        le=22.0,
        description="Faint end of procedural apparent-magnitude range (stars get smaller/dimmer toward this m).",
    )
    band_star_density_scale: float = Field(
        1.68,
        ge=0.5,
        le=8.0,
        description="Multiplier on resolved background/mid star catalog count in galaxy_view.",
    )
    cosmic_star_enabled: bool = Field(
        True,
        description="Population A: isotropic background field (independent of galactic morphology).",
    )
    cosmic_star_density_scale: float = Field(
        1.0,
        ge=0.2,
        le=2.5,
        description="Density scale for isotropic cosmic background stars.",
    )
    halo_star_enabled: bool = Field(
        True,
        description="Population B: thick-disk / halo stars with broad latitude bias.",
    )
    halo_star_density_scale: float = Field(
        1.0,
        ge=0.2,
        le=2.5,
        description="Density scale for halo / thick-disk star population.",
    )
    halo_lat_sigma: float = Field(
        0.52,
        ge=0.28,
        le=1.05,
        description="Gaussian latitude sigma for halo stars (broad plane bias).",
    )
    galactic_star_density_fraction: float = Field(
        0.56,
        ge=0.12,
        le=1.0,
        description=(
            "Fraction of morphology-linked galactic catalog density when cosmic/halo enabled."
        ),
    )
    cosmic_anchor_count: int = Field(
        6,
        ge=0,
        le=24,
        description="Rare bright anchor stars in the isotropic cosmic field.",
    )
    galactic_anchor_star_count: int = Field(
        28,
        ge=0,
        le=32,
        description="Max ultra-bright outliers in morphology-linked galactic catalogs.",
    )
    galactic_overdensity_star_count: int = Field(
        420,
        ge=0,
        le=4000,
        description="Extra resolved stars placed on morphology density peaks.",
    )
    foreground_star_density_scale: float = Field(
        0.68,
        ge=0.35,
        le=1.5,
        description="Scale foreground resolved-star count (~0.65 = 35% fewer; preserves depth separation).",
    )
    unresolved_background_strength: float = Field(
        1.75,
        ge=0.5,
        le=3.0,
        description="Unresolved speckle/integrated-light grain under resolved stars (higher = denser dim field).",
    )
    unresolved_gas_occlusion: float = Field(
        0.36,
        ge=0.0,
        le=1.0,
        description=(
            "Mute unresolved speckle where diffuse gas/emission is thick (grain behind ISM, not through it)."
        ),
    )
    resolved_faint_mag_floor: float = Field(
        13.4,
        ge=9.0,
        le=16.0,
        description="Apparent mag above which resolved stars are increasingly culled (bright stars kept).",
    )
    resolved_faint_dropout: float = Field(
        0.58,
        ge=0.0,
        le=0.95,
        description="Strength of faint resolved-star culling (0=off; frees depth for unresolved background).",
    )
    magnitude_log_slope: float = Field(
        0.6,
        ge=0.2,
        le=1.2,
        description="Uniform–sky style dN/dm ∝ 10^(alpha*m): larger alpha shifts count toward faint magnitudes.",
    )
    magnitude_ref_mag: float = Field(
        9.35,
        description="Reference V-like magnitude mapped to nominal PSF flux=1 inside the HDR stack.",
    )
    max_ultra_bright_stars: int = Field(
        10,
        ge=0,
        le=512,
        description="Cap counts with apparent m brighter than magnitude_ultra_cut (after sampling trim).",
    )
    magnitude_ultra_cut: float = Field(
        6.5,
        description="Ultra-bright procedural stars are clamped/downsampled beyond `max_ultra_bright_stars`.",
    )
    psf_sigma_core_min: float = Field(0.42, ge=0.25, le=3.0)
    psf_sigma_core_max: float = Field(2.05, ge=0.5, le=5.0)
    psf_beta_default: float = Field(
        3.0,
        ge=2.5,
        le=4.0,
        description="Moffat β for stellar wings (2.5–4 typical).",
    )
    psf_fwhm_base_px: float = Field(
        1.32,
        ge=0.8,
        le=6.0,
        description="Base PSF FWHM (px) scaled per-star by fwhm lottery (not brightness alone).",
    )
    psf_fwhm_ref_mag: float = Field(
        8.0,
        ge=4.0,
        le=12.0,
        description="M₀ in FWHM = base_fwhm * (1 + coeff * max(0, M₀ − mag)).",
    )
    psf_fwhm_mag_coeff: float = Field(
        0.06,
        ge=0.0,
        le=0.5,
        description="Weak FWHM nudge per magnitude brighter than M₀ (brightness mostly in flux).",
    )
    psf_fwhm_bright_jitter: float = Field(
        0.16,
        ge=0.0,
        le=0.35,
        description="Log-normal σ on FWHM for bright-tail stars (size variance).",
    )
    psf_fwhm_faint_jitter: float = Field(
        0.04,
        ge=0.0,
        le=0.12,
        description="Uniform FWHM jitter half-width for faint stars (smoother integrated field).",
    )
    psf_halo_sigma_bright: float = Field(1.6, ge=1.5, le=12.0)
    psf_halo_amp: float = Field(0.012, ge=0.0, le=0.5)
    psf_max_half_px: int = Field(28, ge=8, le=72)
    sensor_aperture_mm: float = Field(
        50.0,
        ge=8.0,
        le=200.0,
        description="Effective aperture (mm) for PSF scale (with focal length).",
    )
    sensor_focal_mm: float = Field(
        85.0,
        ge=14.0,
        le=600.0,
        description="Focal length (mm) paired with aperture for f/# PSF scaling.",
    )
    volumetric_g_forward: float = Field(
        0.68,
        ge=-0.1,
        le=0.95,
        description="Henyey–Greenstein g for forward nebula/dust scatter lobe.",
    )
    volumetric_g_back: float = Field(
        -0.32,
        ge=-0.95,
        le=0.15,
        description="Henyey–Greenstein g for backscatter lobe.",
    )
    lf_power_slope: float = Field(
        0.52,
        ge=0.2,
        le=1.0,
        description="dN/dm power-law slope blended with IMF apparent magnitudes.",
    )
    star_reddening_strength: float = Field(
        1.0,
        ge=0.0,
        le=2.5,
        description="Scale for per-star dust reddening before PSF stamp (0=off).",
    )
    bulge_warmth_strength: float = Field(
        0.52,
        ge=0.0,
        le=2.0,
        description="Warmth/flux boost for stars in the galactic bulge region.",
    )
    lane_contrast_amp: float = Field(
        0.24,
        ge=0.0,
        le=0.4,
        description="Multiscale contrast on dark dust lanes after nebula composite.",
    )
    galaxy_tone_curve: Literal["reinhard", "asinh", "filmic", "aces", "acescct"] = Field(
        "asinh",
        description="HDR roll-off toward display for galaxy_view.",
    )
    asinh_stretch_gain: float = Field(
        0.045,
        ge=0.01,
        le=0.5,
        description="Linear canvas asinh gain (≈0.02–0.2); tune with asinh_stretch_q by eye.",
    )
    asinh_stretch_q: float = Field(
        1.22,
        ge=0.35,
        le=2.5,
        description="Linear canvas asinh Q (≈0.5–1.5).",
    )
    disk_asinh_curvature: float = Field(
        0.70,
        ge=0.35,
        le=1.2,
        description="Scales effective asinh gain on disk (<1 = milder, preserves faint arms).",
    )
    asinh_midtone_exposure: float = Field(
        1.0,
        ge=0.7,
        le=1.6,
        description="Mid-tone exposure lift before asinh stretch.",
    )
    asinh_toe_strength: float = Field(
        0.22,
        ge=0.0,
        le=0.65,
        description="Shadow toe softness after asinh (higher = lifted blacks).",
    )
    acescct_grade_strength: float = Field(
        0.62,
        ge=0.0,
        le=1.2,
        description="ACEScct log-grade strength when galaxy_tone_curve=acescct.",
    )
    acescct_shoulder: float = Field(
        0.12,
        ge=0.0,
        le=0.35,
        description="Shoulder compression in ACEScct rolloff passes.",
    )
    final_acescct_rolloff: float = Field(
        0.14,
        ge=0.0,
        le=1.0,
        description="Optional ACEScct cinematic pass after display grade (0=off).",
    )
    isp_chain_strength: float = Field(
        0.52,
        ge=0.0,
        le=1.5,
        description="ISP after linear noise on nebula canvas (0=off; lower preserves star chroma when split).",
    )
    lens_flare_thin_film_strength: float = Field(
        0.034,
        ge=0.0,
        le=0.12,
        description="Thin-film interference tint on lens flare / hot cores.",
    )
    jpeg_highlight_smooth: float = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Post-JPEG low-pass on small-scale highlights (anti-blocking).",
    )
    filmic_shoulder: float = Field(
        0.85,
        ge=0.1,
        le=4.0,
        description="Shoulder strength for galaxy_tone_curve=filmic.",
    )
    faint_unsharp_sigma_px: float = Field(
        1.85,
        ge=0.0,
        le=12.0,
        description="0 disables faint-only micro-contrast after tone map; Gaussian sigma in px (separable approximation).",
    )
    faint_unsharp_amp: float = Field(
        0.11,
        ge=0.0,
        le=0.5,
        description="Strength of faint unsharp blend (applied only below luma knee).",
    )
    faint_unsharp_luma_knee: float = Field(
        0.065,
        ge=0.0,
        le=1.0,
        description="Only pixels with linear luma below this knee pick up faint unsharp (post disk-weighted gamma).",
    )
    sensor_noise_stage: Literal["off", "linear", "display", "both"] = Field(
        "both",
        description="Poisson/read noise: linear=pre tone-map; display=post tone-map camera pipeline.",
    )
    sensor_shot_noise_scale: float = Field(
        0.0018,
        ge=0.0,
        le=4.0,
        description="Shot-noise scale in linear scene space (before tone map).",
    )
    sensor_read_noise_sigma: float = Field(
        0.00055,
        ge=0.0,
        le=0.05,
        description="Read-noise sigma in linear scene space.",
    )
    sensor_display_shot_scale: float = Field(
        0.0016,
        ge=0.0,
        le=0.02,
        description="Shot-noise scale after tone map (display / camera pipeline).",
    )
    sensor_display_read_sigma: float = Field(
        0.00042,
        ge=0.0,
        le=0.02,
        description="Read-noise sigma after tone map.",
    )
    halation_strength: float = Field(
        0.024,
        ge=0.0,
        le=0.2,
        description="Wavelength-dependent halation on saturated nebula / bright disk.",
    )
    chromatic_aberration_strength: float = Field(
        0.62,
        ge=0.0,
        le=2.5,
        description="Mild lateral chromatic aberration toward edges (bright cores).",
    )
    star_position_jitter_px: float = Field(
        0.42,
        ge=0.0,
        le=1.5,
        description="Extra sub-pixel PSF stamp jitter (breaks grid aliasing / clumping).",
    )
    blue_noise_dither_strength: float = Field(
        0.72,
        ge=0.0,
        le=2.0,
        description="Blue-noise dither amplitude before 8-bit save (0=off).",
    )
    nebula_fine_noise_mix: float = Field(
        0.66,
        ge=0.35,
        le=1.0,
        description="Weight of fine FBM in galaxy nebula stack (lower = less speckle).",
    )
    disk_star_density_dropout: float = Field(
        0.38,
        ge=0.0,
        le=0.62,
        description="Prob. to skip bg/mid stars in bright disk (scaled by disk_w; fg unchanged).",
    )
    star_flux_scatter_sigma: float = Field(
        0.08,
        ge=0.0,
        le=0.18,
        description="Per-star flux multiplier scatter: 1 + N(0, σ) for photoreal catalog.",
    )
    faint_star_chroma_desat: float = Field(
        0.09,
        ge=0.0,
        le=0.28,
        description="Desaturate faint star display chroma so the band reads as integrated light.",
    )
    star_composite_add_scale: float = Field(
        0.92,
        ge=0.5,
        le=1.0,
        description="Scale star layer before add+max onto graded canvas (lower avoids blowing disk).",
    )
    star_band_chroma_desat: float = Field(
        0.06,
        ge=0.0,
        le=0.45,
        description="Extra chroma desat for stars on the bright galactic plane (integrated light).",
    )
    star_band_brightness_scale: float = Field(
        0.88,
        ge=0.35,
        le=1.0,
        description="Star brightness multiplier on the bright plane (lower = less blue fringe).",
    )
    star_band_chroma_adapt: float = Field(
        0.12,
        ge=0.0,
        le=0.65,
        description="Pull star chroma toward local graded canvas in the plane (combination fix).",
    )
    nebula_emit_scale_with_stars: float = Field(
        0.62,
        ge=0.25,
        le=1.0,
        description="Scale H II / line emission when catalog stars composite above nebula.",
    )
    star_display_white_cap: float = Field(
        0.96,
        ge=0.75,
        le=1.0,
        description="Display white cap for star tone map before compositing.",
    )
    depth_of_field_strength: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Gaussian depth blur on star layers (0=sharp stars; nebula canvas separate).",
    )
    star_display_stretch_gain: float = Field(
        0.0,
        ge=0.0,
        le=24.0,
        description="Star percentile-asinh gain (0 = auto: top peak_percentile → display white).",
    )
    star_stretch_peak_percentile: float = Field(
        99.90,
        ge=99.0,
        le=99.95,
        description="HDR star luma percentile mapped to display white before add-max (0.1–0.5% tail).",
    )
    extinction_r_v: float = Field(
        3.1,
        ge=2.5,
        le=5.5,
        description="CCM total-to-selective extinction ratio R_V (MW-like ≈ 3.1).",
    )
    extinction_lane_mag_min: float = Field(
        0.5,
        ge=0.1,
        le=1.5,
        description="Target lane extinction at moderate dust (mag).",
    )
    extinction_lane_mag_max: float = Field(
        3.1,
        ge=1.0,
        le=4.0,
        description="Target extinction in darkest lanes (mag).",
    )
    extinction_transmission_floor: float = Field(
        0.06,
        ge=0.02,
        le=0.35,
        description="Transmission floor used to calibrate lane A_V scale.",
    )
    display_black_point: float = Field(0.045, ge=0.0, le=0.15)
    display_white_point: float = Field(0.98, ge=0.75, le=1.0)
    sky_darken_strength: float = Field(
        0.58,
        ge=0.0,
        le=0.85,
        description="Crush sky latitudes toward black after compositing stars.",
    )
    depth_of_field_max_px: float = Field(
        3.2,
        ge=0.5,
        le=14.0,
        description="Max blur radius (px) for far star layer at ~2K width.",
    )
    depth_blur_nebula: bool = Field(
        False,
        description="If true, also blur nebula canvas (often smears emission into noise).",
    )
    filmic_s_curve_strength: float = Field(
        0.12,
        ge=0.0,
        le=0.6,
        description="Mild global S-curve in display finish only (core depth uses localized dodge/burn).",
    )
    split_toning_strength: float = Field(
        0.032,
        ge=0.0,
        le=0.5,
        description="Subtle warm highlights / cool shadows (display finish).",
    )
    neutral_wb_strength: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Gray-world white balance toward neutral before split-tone.",
    )
    local_contrast_strength: float = Field(
        0.14,
        ge=0.0,
        le=0.35,
        description="Modest CLAHE-like local contrast (0=off).",
    )
    display_bloom_strength: float = Field(
        0.034,
        ge=0.0,
        le=0.5,
        description="Three-scale thresholded bloom on bright disk (nebula canvas when split_star_display_grade).",
    )
    bloom_threshold: float = Field(
        0.94,
        ge=0.5,
        le=0.99,
        description="Display-luma threshold for bloom (higher = tighter hot core only).",
    )
    bloom_mix_tight: float = Field(0.52, ge=0.0, le=1.0)
    bloom_mix_mid: float = Field(0.22, ge=0.0, le=1.0)
    bloom_mix_wide: float = Field(0.06, ge=0.0, le=1.0)
    god_rays_strength: float = Field(0.0, ge=0.0, le=0.4)
    core_burn_strength: float = Field(0.10, ge=0.0, le=0.5)
    core_dodge_strength: float = Field(
        0.035,
        ge=0.0,
        le=0.25,
        description="Localized rim dodge on bulge core (small radius, not global contrast).",
    )
    core_local_scurve_strength: float = Field(
        0.20,
        ge=0.0,
        le=0.5,
        description="Localized S-curve on nebula core for perceived depth.",
    )
    core_local_scurve_radius_passes: int = Field(
        2,
        ge=1,
        le=6,
        description="Blur passes defining localized core grade radius (~2–3 px at 2K).",
    )
    core_clahe_strength: float = Field(0.0, ge=0.0, le=0.35)
    lightwrap_strength: float = Field(0.03, ge=0.0, le=0.35)
    cluster_placement_strength: float = Field(
        0.72,
        ge=0.0,
        le=1.5,
        description="Worley/FBM cluster mask weight for star density.",
    )
    motion_blur_strength: float = Field(
        0.0,
        ge=0.0,
        le=0.25,
        description="Subtle directional blur on background stars (galactic rotation).",
    )
    color_stratify_strength: float = Field(
        0.07,
        ge=0.0,
        le=0.4,
        description="Warm core / cool halo color gradient (stars + mild canvas).",
    )
    shadow_lift: float = Field(0.024, ge=0.0, le=0.06)
    highlight_chroma_desat: float = Field(
        0.04,
        ge=0.0,
        le=0.14,
        description="Desaturate top display highlights (anti-neon tail).",
    )
    star_peak_soft_clamp: float = Field(
        0.18,
        ge=0.0,
        le=0.65,
        description="Soft knee on star peaks before add-max over nebula canvas.",
    )
    star_midlayer_scale: float = Field(
        0.36,
        ge=0.0,
        le=1.0,
        description="Fractional mid-depth star layer (0=background+foreground only).",
    )
    placement_asymmetry: float = Field(
        0.18,
        ge=0.0,
        le=0.35,
        description="Break mirror symmetry in star placement (lon/lat warp).",
    )
    morphology_void_strength: float = Field(
        0.92,
        ge=0.0,
        le=1.0,
        description="Local emptiness carved into stellar density (ugly voids).",
    )
    morphology_scar_strength: float = Field(
        0.78,
        ge=0.0,
        le=1.5,
        description="Ridged dust scars in morphology absorption map.",
    )
    morphology_discontinuity_cut_strength: float = Field(
        0.78,
        ge=0.0,
        le=1.2,
        description="Apply disruption gouges to stellar density G (not just extinction).",
    )
    morphology_macro_void_count: int = Field(
        3,
        ge=0,
        le=8,
        description="Rare torn voids along the plane carved into G.",
    )
    morphology_dust_primary: bool = Field(
        True,
        description="Morphology absorption min before ext blur; skip horizontal ext smear.",
    )
    unresolved_deposit_primary: bool = Field(
        True,
        description="Unresolved speckle from catalog deposit pyramid (not smooth U grain).",
    )
    morphology_seed_perturb_scale: float = Field(
        1.18,
        ge=0.0,
        le=2.0,
        description="Master scale for per-seed ε on cluster / SF / dust morphology maps (0=off).",
    )
    morphology_sf_perturb: float = Field(
        0.16,
        ge=0.0,
        le=0.45,
        description="Local star-formation efficiency ε amplitude (multiplicative on SF map).",
    )
    morphology_dust_perturb: float = Field(
        0.18,
        ge=0.0,
        le=0.45,
        description="Local dust-density ε amplitude (multiplicative on absorption map).",
    )
    morphology_dust_micro_strength: float = Field(
        1.25,
        ge=0.0,
        le=1.5,
        description="Independent small/fine dust scales on morphology absorption (dust_D detail).",
    )
    morphology_cluster_perturb: float = Field(
        0.22,
        ge=0.0,
        le=0.55,
        description="Local cluster-likelihood ε amplitude (multiplicative on cluster map).",
    )
    morphology_absorption_contrast: float = Field(
        2.05,
        ge=0.8,
        le=2.2,
        description="Percentile contrast on morphology dust absorption (higher = nastier lane mask).",
    )
    morphology_extinction_strength: float = Field(
        2.15,
        ge=0.6,
        le=2.5,
        description="Scale morphology absorption before transmission (cloud dominance).",
    )
    morphology_extinction_av_scale: float = Field(
        1.38,
        ge=0.8,
        le=2.5,
        description="Extra A_V on canvas when applying morphology extinction to unresolved light.",
    )
    morphology_extinction_fill_suppress: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Reduce dust backlight/rim/mottle fill after extinction (0=full legacy fill).",
    )
    morphology_nebula_haze_scale: float = Field(
        0.52,
        ge=0.05,
        le=2.0,
        description="Scale nebula haze/glow/band additives when morphology dust is primary.",
    )
    morphology_nebula_gas_scale: float = Field(
        0.88,
        ge=0.15,
        le=3.5,
        description="Procedural nebula gas weight when morphology dust is primary (lower = morphology wins).",
    )
    morphology_continuum_suppress: float = Field(
        0.58,
        ge=0.0,
        le=1.0,
        description=(
            "Scale smooth nebula-derived haze/cloud/glow when morphology is primary "
            "(seed-independent blurs that otherwise hide ISM hierarchy)."
        ),
    )
    morphology_nebula_emit_scale: float = Field(
        1.28,
        ge=0.4,
        le=3.5,
        description="H II / emission-line layer multiplier when morphology dust is primary.",
    )
    morphology_nebula_emit_gate_power: float = Field(
        1.88,
        ge=0.8,
        le=3.0,
        description="Emission-only extinction gate exponent (gas/haze use a milder mask).",
    )
    morphology_extinction_lane_carve: float = Field(
        0.88,
        ge=0.0,
        le=0.95,
        description="Final asymmetric lane carve on morphology extinction map.",
    )
    morphology_lane_fragment_strength: float = Field(
        0.74,
        ge=0.0,
        le=1.0,
        description="Roughen filament lanes (edge texture, not multiplicative hole punching).",
    )
    morphology_missing_region_boost: float = Field(
        0.44,
        ge=0.0,
        le=0.95,
        description="Second-pass canvas mask in darkest extinction lanes (missing regions).",
    )
    render_profile: RenderProfile = Field(
        RenderProfile.full,
        description="physical=linear only; physical_grade=+linear grade; full=display finish.",
    )
    debug_export_layers: bool = Field(
        False,
        description="Write morphology/star/extinction layer PNGs alongside main output.",
    )
    debug_grayscale_morphology: bool = Field(
        False,
        description="Write grayscale morphology diagnostic (G + 1-D + U).",
    )
    use_imf_magnitudes: bool = Field(
        True,
        description="Sample apparent mags from Salpeter IMF + distance modulus (vs log power-law only).",
    )
    imf_giant_fraction: float = Field(
        0.055,
        ge=0.0,
        le=0.2,
        description="Fraction of field stars drawn from the bright-giant IMF tail.",
    )
    imf_cluster_giant_fraction: float = Field(
        0.11,
        ge=0.0,
        le=0.35,
        description="Giant-tail fraction inside association/cluster catalog.",
    )
    vignette_strength: float = Field(
        1.0,
        ge=0.0,
        le=2.5,
        description="Scale long-exposure vignette when galaxy_view.",
    )
    film_grain_strength: float = Field(
        0.04,
        ge=0.0,
        le=0.15,
        description="Blue-noise film grain RMS at display (galaxy_view); 0=off.",
    )
    emission_unsharp_amount: float = Field(
        0.14,
        ge=0.0,
        le=0.22,
        description="Masked high-pass on emission before composite (clump micro-contrast).",
    )
    disk_height: float = Field(0.22, ge=0.06, le=0.45, description="sech² disk scale height in lat radians.")
    halo_fraction: float = Field(0.22, ge=0.0, le=0.65, description="Weight of halo vs disk star population.")
    halo_power: float = Field(1.35, ge=0.5, le=2.5, description="Halo density power-law exponent in |lat|.")
    poisson_min_sep_bright_px: float = Field(14.0, ge=4.0, le=48.0)
    poisson_min_sep_faint_px: float = Field(3.0, ge=1.0, le=12.0)
    mag_radius_gamma: float = Field(
        0.10,
        ge=0.0,
        le=0.65,
        description="Weak PSF radius ∝ flux^gamma (brightness mostly flux, not footprint).",
    )
    m_bloom_cut: float = Field(7.8, ge=5.0, le=14.0, description="Stars brighter than this get bloom layer.")
    spike_max_count: int = Field(4, ge=0, le=64)
    band_lat_sigma: float = Field(0.135, ge=0.04, le=0.22)
    band_rotation_deg: float = Field(2.5, ge=-12.0, le=12.0)
    band_curvature_amp: float = Field(
        0.078,
        ge=0.0,
        le=0.14,
        description="S/U/W midplane warp amplitude (longitude sine family).",
    )
    band_thickness_asymmetry: float = Field(
        0.42,
        ge=0.0,
        le=0.72,
        description="How much thinner vs thicker above/below the warped midplane (seed-locked).",
    )
    disk_mesoscale_thickness_strength: float = Field(
        0.58,
        ge=0.0,
        le=1.0,
        description="Mesoscale puff/compress/evacuate disk scale height (1+field), not holes.",
    )
    bulge_n: float = Field(1.4, ge=0.5, le=6.0, description="Sérsic index for central bulge layer.")
    bulge_scale: float = Field(0.22, ge=0.04, le=0.35, description="Bulge extent along galactic longitude (wider = softer core).")
    bulge_intensity: float = Field(0.08, ge=0.0, le=1.5)
    bulge_desat: float = Field(0.35, ge=0.0, le=1.0)
    midplane_unsharp_amp: float = Field(0.09, ge=0.0, le=0.35)
    nebula_color_strength: float = Field(1.0, ge=0.3, le=2.0)
    dust_opacity: float = Field(1.0, ge=0.3, le=2.0, description="Scale dust lane carve strength.")
    catalog_mode: Literal["off", "positions", "stats_seed", "luminance_overlay"] = Field(
        "off",
        description="Optional real-catalog anchoring.",
    )
    catalog_blend: float = Field(0.15, ge=0.0, le=1.0)
    luminance_map_path: Path | None = Field(None, description="Optional Milky Way luminance overlay image.")
    aces_exposure: float = Field(1.0, ge=0.25, le=4.0)
    use_spectral_teffective: bool = Field(
        True,
        description="Sample OBAFGKM Teff for star colors (else legacy B–V locus).",
    )


class NebulaTuningConfig(BaseModel):
    """Artist-facing controls for nebula structure and dust extinction.

    Tuning guide (keep this updated as renderer behavior evolves):
    - Out-of-box baseline: use the defaults in this model.
    - Fast first pass: adjust only `style`; then touch one scalar at a time.
    - `cloud_continuity`: higher = more connected/thicker cloud bands.
    - `dust_coverage`: higher = dust appears in more places.
    - `dust_strength`: higher = stronger star/background dimming in dust.
    Recommended fine-tune ranges after style selection:
    - cloud_continuity: 0.9-1.25
    - dust_coverage: 0.85-1.15
    - dust_strength: 0.80-1.20
    """

    style: Literal["subtle", "balanced", "dramatic"] = Field(
        "balanced",
        description="High-level nebula preset controlling dust/extinction character; start here before scalar tweaks.",
    )
    cloud_continuity: float = Field(
        1.22,
        ge=0.6,
        le=1.6,
        description="Cloud connectivity/thickness along the galactic band; higher values produce denser connected volumes.",
    )
    dust_coverage: float = Field(
        1.06,
        ge=0.5,
        le=1.6,
        description="Fraction of the band eligible for dark dust/void carving; lower values create fewer dust regions.",
    )
    dust_strength: float = Field(
        1.22,
        ge=0.5,
        le=1.8,
        description="Dust extinction strength against stars/background; raise only after coverage feels right.",
    )
    emit_halpha_saturation: float = Field(
        0.68,
        ge=0.2,
        le=1.0,
        description="H-alpha pink/red saturation (lower = subtler).",
    )
    emit_patch_strength: float = Field(
        1.05,
        ge=0.4,
        le=2.5,
        description="Scale localized H II red/magenta cloud patches along the band.",
    )
    debug_pass: NebulaDebugPass = Field(
        "normal",
        description=(
            "Diagnostic render for galaxy_streak nebula: `normal` full composite; `occluder_only` shows "
            "dust_rgb blend only; `continuum_only` shows synthetic low-frequency continuum only (boosted for visibility). "
            "`layer_*` visualize procedural noise layers (grayscale nebula); `layer_carve` / `layer_dust_alpha` show "
            "carve and combined dust alpha; `mask_only` shows latitudinal band mask; `warp_vectors` false-colors warp X/Y as R/G."
        ),
    )


class RenderConfig(BaseModel):
    width: int = Field(2048, ge=256, le=16384)
    height: int = Field(1024, ge=128, le=8192)
    output_base_name: str = "starsky"
    output_dir: Path = Path("output")
    generations: int = Field(1, ge=1, le=1000)
    seed: int | None = None
    projection_mode: ProjectionMode = ProjectionMode.both
    output_format: OutputFormat = OutputFormat.png
    nebula_mode: NebulaMode = NebulaMode.galaxy_streak
    quality: int = Field(100, ge=50, le=100)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    nebula_tuning: NebulaTuningConfig = Field(
        default_factory=NebulaTuningConfig,
        description="Procedural nebula tuning block used by renderer and config files.",
    )
    cubemap_face_size: int = Field(1024, ge=128, le=4096)
    wrap_safe: Literal[True] = True

    @field_validator("output_base_name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip().replace(" ", "_")
        if not cleaned:
            raise ValueError("output_base_name cannot be empty")
        return cleaned
