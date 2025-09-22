from __future__ import annotations
import torch


class LinRGB:
    @staticmethod
    def from_sRGB(image: torch.Tensor) -> torch.Tensor:
        # Convert sRGB to linearRGB
        # (copied from daltonlens.convert.sRGB_from_linearRGB)
        out = torch.empty_like(image)
        small_mask = image < 0.04045
        large_mask = torch.logical_not(small_mask)
        out[small_mask] = image[small_mask] / 12.92
        out[large_mask] = torch.pow((image[large_mask] + 0.055) / 1.055, 2.4)
        return out

    @staticmethod
    def to_sRGB(image: torch.Tensor) -> torch.Tensor:
        # Convert linearRGB to sRGB.
        # Made on the basis of daltonlens.convert.sRGB_from_linearRGB
        # by Nicolas Burrus. Clipping operation was removed.
        out = torch.empty_like(image)
        small_mask = image < 0.0031308
        large_mask = torch.logical_not(small_mask)
        out[small_mask] = image[small_mask] * 12.92
        out[large_mask] = (
            torch.pow(image[large_mask], 1.0 / 2.4) * 1.055 - 0.055
        )
        return out
