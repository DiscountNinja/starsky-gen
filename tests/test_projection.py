import numpy as np

from starsky_gen.projections import cubemap_faces_from_equirect


def test_equirect_wrap_continuity() -> None:
    h, w = 256, 512
    img = np.zeros((h, w, 3), dtype=np.float64)
    row = np.linspace(0.0, 1.0, h)[:, None]
    img[:, :, 0] = row
    img[:, :, 1] = 0.2
    img[:, :, 2] = 0.5
    img[:, 0] = img[:, -1]
    diff = np.abs(img[:, 0] - img[:, -1]).mean()
    assert diff < 1e-9


def test_cubemap_faces_generate_expected_shape() -> None:
    h, w = 128, 256
    img = np.random.default_rng(2).random((h, w, 3))
    faces = cubemap_faces_from_equirect(img, face_size=32)
    assert set(faces.keys()) == {"px", "nx", "py", "ny", "pz", "nz"}
    for face in faces.values():
        assert face.shape == (32, 32, 3)
