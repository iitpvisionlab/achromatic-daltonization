from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Literal

from .cs import Lab, Prolab


class CD:
    def __init__(
        self, cs: Literal["lab", "prolab"], lightness_weight: float = 0
    ) -> None:
        if cs == "lab":
            self.transform = Lab.from_sRGB
        elif cs == "prolab":
            self.transform = self._to_prolab
        self.lightness_weight = lightness_weight

    def _to_prolab(self, image: npt.NDArray):
        lab = Prolab.from_sRGB(image)
        lab[:, :, 1] /= np.maximum(lab[:, :, 0], 1e-5)
        lab[:, :, 2] /= np.maximum(lab[:, :, 0], 1e-5)
        return lab

    def __call__(
        self,
        image1: npt.NDArray,
        image2: npt.NDArray,
    ) -> float:
        lab1 = self.transform(image1)
        lab2 = self.transform(image2)
        diff = lab1 - lab2
        weights = diff * np.array([np.sqrt(self.lightness_weight), 1, 1])
        return np.mean(np.linalg.norm(weights, axis=2))


class RMS:
    def __init__(
        self,
        cs: Literal["lab", "prolab"],
        n_neighbors: int = 1000,
        step: int = 10,
        sigma_rate: float = 0.25,
    ) -> None:
        if cs == "lab":
            self.transform = Lab.from_sRGB
        elif cs == "prolab":
            self.transform = Prolab.from_sRGB

        self.n_neighbors = n_neighbors
        self.step = step
        self.sigma_rate = sigma_rate

    def generate_random_neighbors(self, size: tuple[int, int], seed: int):
        sigma = [s * self.sigma_rate for s in size]
        indices = np.stack(self.indices, axis=-1)
        rng = np.random.default_rng(seed)
        return (
            rng.normal(
                indices,
                sigma,
                (self.n_neighbors, self.dst_height, self.dst_width, 2),
            )
            .round()
            .astype(int)
        )

    def pixel_contrasts(
        self,
        image: npt.NDArray,
        cy: npt.NDArray,
        cx: npt.NDArray,
        ny: npt.NDArray,
        nx: npt.NDArray,
    ) -> npt.NDArray:
        return np.linalg.norm(image[cy, cx, :] - image[ny, nx, :], axis=-1)

    def __call__(self, image1: npt.NDArray, image2: npt.NDArray) -> float:
        h, w, _ = image1.shape
        self.dst_height, self.dst_width = h // self.step, w // self.step
        h_indices = np.arange(self.dst_height) * self.step
        w_indices = np.arange(self.dst_width) * self.step
        self.indices = np.meshgrid(h_indices, w_indices, indexing="ij")
        grid_h, grid_w = self.indices
        cy = np.broadcast_to(
            grid_h[np.newaxis, ...],
            (self.n_neighbors, self.dst_height, self.dst_width),
        )
        cx = np.broadcast_to(
            grid_w[np.newaxis, ...],
            (self.n_neighbors, self.dst_height, self.dst_width),
        )

        seed = hash(np.mean(image1 + image2))

        neighbors = self.generate_random_neighbors((h, w), seed)

        valid_mask = (
            (0 <= neighbors[..., 0])
            & (neighbors[..., 0] < h)
            & (0 <= neighbors[..., 1])
            & (neighbors[..., 1] < w)
        )

        ny = neighbors[..., 0].clip(0, h - 1)
        nx = neighbors[..., 1].clip(0, w - 1)

        lab1 = self.transform(image1)
        lab2 = self.transform(image2)
        image1_contrast = self.pixel_contrasts(lab1, cy, cx, ny, nx)
        image2_contrast = self.pixel_contrasts(lab2, cy, cx, ny, nx)

        contrast_diff = (image1_contrast - image2_contrast) / 160
        mean = (contrast_diff**2 * valid_mask).sum(axis=0) / valid_mask.sum(
            axis=0
        )
        return np.mean(np.sqrt(mean))
