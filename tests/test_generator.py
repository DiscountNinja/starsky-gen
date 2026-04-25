from pathlib import Path

import numpy as np
from PIL import Image

from starsky_gen.config import FeatureConfig, OutputFormat, ProjectionMode, RenderConfig
from starsky_gen.generator import render_single


def test_seed_reproducibility(tmp_path: Path) -> None:
    cfg = RenderConfig(
        width=512,
        height=256,
        output_base_name="seed",
        output_dir=tmp_path,
        generations=1,
        seed=100,
        projection_mode=ProjectionMode.equirectangular,
        output_format=OutputFormat.png,
        features=FeatureConfig(jpeg_artifact_pass=False),
    )

    out_a, _ = render_single(cfg, 0)
    out_b, _ = render_single(cfg, 0)
    a = np.asarray(Image.open(out_a["equirectangular"]).convert("RGB"))
    b = np.asarray(Image.open(out_b["equirectangular"]).convert("RGB"))
    assert np.array_equal(a, b)
