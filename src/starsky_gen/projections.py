from __future__ import annotations

import numpy as np


def sph_to_equirect_xy(
    lon: np.ndarray,
    lat: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = (lon / (2.0 * np.pi) * width).astype(np.int32) % width
    y = ((0.5 - (lat / np.pi)) * height).astype(np.int32)
    return x, np.clip(y, 0, height - 1)


def sph_to_equirect_xy_float(
    lon: np.ndarray | float,
    lat: np.ndarray | float,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fractional equirectangular coordinates prior to quantization (supports bilinear lookups)."""
    lon = np.asanyarray(lon, dtype=np.float64)
    lat = np.asanyarray(lat, dtype=np.float64)
    xf = np.mod(lon / (2.0 * np.pi) * float(width), float(width))
    yf = (0.5 - (lat / np.pi)) * float(height)
    yf = np.clip(yf, 0.0, float(height) - 1.0 - 1e-9)
    return xf.astype(np.float64), yf.astype(np.float64)


def cubemap_uv_to_dir(face: str, u: float, v: float) -> np.ndarray:
    if face == "px":
        vec = np.array([1.0, v, -u], dtype=np.float64)
    elif face == "nx":
        vec = np.array([-1.0, v, u], dtype=np.float64)
    elif face == "py":
        vec = np.array([u, 1.0, -v], dtype=np.float64)
    elif face == "ny":
        vec = np.array([u, -1.0, v], dtype=np.float64)
    elif face == "pz":
        vec = np.array([u, v, 1.0], dtype=np.float64)
    else:
        vec = np.array([-u, v, -1.0], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def dir_to_lon_lat(direction: np.ndarray) -> tuple[float, float]:
    x, y, z = direction
    lon = np.arctan2(z, x) % (2.0 * np.pi)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    return lon, lat


def sample_equirect(equirect: np.ndarray, lon: float, lat: float) -> np.ndarray:
    h, w, _ = equirect.shape
    x = int((lon / (2.0 * np.pi) * w) % w)
    y = int(np.clip((0.5 - lat / np.pi) * h, 0, h - 1))
    return equirect[y, x]


def cubemap_faces_from_equirect(equirect: np.ndarray, face_size: int) -> dict[str, np.ndarray]:
    faces = {}
    for face in ["px", "nx", "py", "ny", "pz", "nz"]:
        out = np.zeros((face_size, face_size, 3), dtype=np.float64)
        for j in range(face_size):
            v = 1.0 - 2.0 * ((j + 0.5) / face_size)
            for i in range(face_size):
                u = 2.0 * ((i + 0.5) / face_size) - 1.0
                direction = cubemap_uv_to_dir(face, u, v)
                lon, lat = dir_to_lon_lat(direction)
                out[j, i] = sample_equirect(equirect, lon, lat)
        faces[face] = out
    return faces
