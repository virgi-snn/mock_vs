from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class Volume:
    data: np.ndarray
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def copy(self) -> "Volume":
        return Volume(self.data.copy(), self.spacing)


def gaussian1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def convolve1d_axis(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (pad, pad)
    a = np.pad(arr, pads, mode="edge")
    out = np.empty_like(arr, dtype=np.float32)
    it = np.nditer(out, flags=["multi_index"], op_flags=["writeonly"])
    while not it.finished:
        idx = list(it.multi_index)
        s = 0.0
        for k in range(kernel.size):
            idx_p = idx.copy()
            idx_p[axis] = idx[axis] + k
            s += kernel[k] * a[tuple(idx_p)]
        it[0] = s
        it.iternext()
    return out


def convolve_separable3d(vol: Volume, sigma: float) -> Volume:
    k = gaussian1d(sigma, int(3 * max(1.0, sigma)))
    d = vol.data.astype(np.float32, copy=False)
    d = convolve1d_axis(d, k, axis=0)
    d = convolve1d_axis(d, k, axis=1)
    d = convolve1d_axis(d, k, axis=2)
    return Volume(d, vol.spacing)


def synthetic_spheres(
    shape: Tuple[int, int, int],
    centers: List[Tuple[float, float, float]],
    radii: List[float],
    noise: float = 0.1,
) -> Volume:
    z, y, x = np.indices(shape).astype(np.float32)
    vol = np.zeros(shape, dtype=np.float32)
    for (cz, cy, cx), r in zip(centers, radii):
        dist2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
        vol += (dist2 <= r * r).astype(np.float32)
    vol += noise * np.random.RandomState(42).normal(size=shape)
    vol = np.clip(vol, 0.0, None)
    return Volume(vol)


class PipelineStep:
    def __call__(self, vol: Volume) -> Volume:
        raise NotImplementedError


class DenoiseStep(PipelineStep):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, vol: Volume) -> Volume:
        return convolve_separable3d(vol, self.sigma)


class GradientStep(PipelineStep):
    def __call__(self, vol: Volume) -> Volume:
        gz, gy, gx = np.gradient(vol.data)
        g = np.sqrt(gz * gz + gy * gy + gx * gx)
        return Volume(g, vol.spacing)


class ThresholdStep(PipelineStep):
    def __init__(self, value: float = 95.0):
        self.value = value

    def __call__(self, vol: Volume) -> Volume:
        d = vol.data
        thr = np.percentile(d, self.value)
        mask = (d >= thr).astype(np.uint8)
        return Volume(mask, vol.spacing)#[0] ####################


class LabelStep(PipelineStep):
    def __call__(self, vol: Volume) -> Volume:
        d = vol.data.astype(np.uint8)
        labels = np.zeros_like(d, dtype=np.int32)
        current = 0
        zmax, ymax, xmax = d.shape
        stack = []
        for z in range(zmax):
            for y in range(ymax):
                for x in range(xmax):
                    if d[z, y, x] and labels[z, y, x] == 0:
                        current += 1
                        stack.append((z, y, x))
                        labels[z, y, x] = current
                        while stack:
                            cz, cy, cx = stack.pop()
                            for dz, dy, dx in (
                                (1, 0, 0),
                                (-1, 0, 0),
                                (0, 1, 0),
                                (0, -1, 0),
                                (0, 0, 1),
                                (0, 0, -1),
                            ):
                                nz, ny, nx = cz + dz, cy + dy, cx + dx
                                if 0 <= nz < zmax and 0 <= ny < ymax and 0 <= nx < xmax:
                                    if d[nz, ny, nx] and labels[nz, ny, nx] == 0:
                                        labels[nz, ny, nx] = current
                                        stack.append((nz, ny, nx))
        return Volume(labels, vol.spacing)


class MeasureStep(PipelineStep):
    def __call__(self, vol: Volume) -> Volume:
        labels = vol.data
        n = int(labels.max())
        table = np.zeros((n, 7), dtype=np.float32)
        for i in range(1, n + 1):
            mask = labels == i
            count = mask.sum()
            if count == 0:
                continue
            idx = np.argwhere(mask)
            cz, cy, cx = idx.mean(axis=0)
            table[i - 1, 0] = i
            table[i - 1, 1] = count
            table[i - 1, 2] = cz
            table[i - 1, 3] = cy
            table[i - 1, 4] = cx
            dz = np.diff(idx[:, 0]).var() if idx.shape[0] > 1 else 0.0
            dy = np.diff(idx[:, 1]).var() if idx.shape[0] > 1 else 0.0
            dx = np.diff(idx[:, 2]).var() if idx.shape[0] > 1 else 0.0
            table[i - 1, 5] = dz + dy + dx
            table[i - 1, 6] = float(count) * (
                vol.spacing [0] * vol.spacing[1] * vol.spacing[2])
        return Volume(table)


class Pipeline:
    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, vol: Volume) -> Volume:
        v = vol
        for s in self.steps:
            v = s(v)
        return v


def normalize(vol: Volume) -> Volume:
    d = vol.data.astype(np.float32, copy=False)
    mn = d.min()
    mx = d.max()
    if mx - mn < 1e-12:
        return Volume(np.zeros_like(d))
    return Volume((d - mn) / (mx - mn), vol.spacing)


def image_processing_pipeline() -> Pipeline:
    steps = [
        DenoiseStep(1.25),
        GradientStep(),
        ThresholdStep(0.95),
        LabelStep(),
        MeasureStep(),
    ]
    return Pipeline(steps)


def generate_volume() -> Volume:
    np.random.seed(0)
    shape = (32, 32, 32)
    centers = [(20.0, 20.0, 20.0), (40.0, 40.0, 44.0), (32.0, 24.0, 48.0)]
    radii = [10.0, 8.0, 6.0]
    v = synthetic_spheres(shape, centers, radii, noise=0.15)
    v = normalize(v)
    return v


def run_demo(pipeline_factory: Callable[[], Pipeline]) -> Dict[str, np.ndarray]:
    v0 = generate_volume()
    p = pipeline_factory()
    out = p.run(v0)
    return {"labels": out.data, "volume": v0.data}


if __name__ == "__main__":
    # I am not working - please fix me!
    # Try to use the debugger tool of your IDE to
    # to trace the bug from the stacktrace line that
    # is crashing the program to the line needs to be
    # modified to fix the bug.

    results = run_demo(image_processing_pipeline)
    print(results)
