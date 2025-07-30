from __future__ import annotations
from typing import Literal
import torch
from .cs import LinRGB


class FarupSimulation:
    cvd_type: Literal["rg", "by"]

    def __init__(self, cvd_type: Literal["rg", "by"], device) -> None:
        self.cvd_type = cvd_type
        if cvd_type == "rg":
            self.matrix = torch.tensor(
                ((0.5, 0.5, 0), (0.5, 0.5, 0), (0.0, 0.0, 1.0))
            ).to(device)
        elif cvd_type == "by":
            self.matrix = torch.tensor(
                ((0.75, -0.25, 0.5), (-0.25, 0.75, 0.5), (0.25, 0.25, 0.5))
            ).to(device)
        else:
            raise KeyError("unknown cvd type:", cvd_type)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.matrix @ image


class VienotSimulation:
    cvd_type: Literal["protan", "deutan"]
    LMS_from_RGB = torch.tensor(
        (
            (0.27293945, 0.66418685, 0.06287371),
            (0.10022701, 0.78761123, 0.11216177),
            (0.01781695, 0.10961952, 0.87256353),
        )
    )
    RGB_from_LMS = torch.tensor(
        (
            (5.30329968, -4.49954803, 0.19624834),
            (-0.67146001, 1.86248629, -0.19102629),
            (-0.0239335, -0.14210614, 1.16603964),
        )
    )
    protan_matrix = torch.tensor(
        (
            (0.0, 1.06481845, -0.06481845),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    deutan_matrix = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.93912723, 0.0, 0.06087277),
            (0.0, 0.0, 1.0),
        )
    )

    def __init__(self, cvd_type: Literal["protan", "deutan"], device) -> None:
        if cvd_type == "protan":
            sim_matrix = self.protan_matrix
        elif cvd_type == "deutan":
            sim_matrix = self.deutan_matrix
        else:
            raise KeyError("unknown cvd type:", cvd_type)

        self.cvd_type = cvd_type

        self.matrix = self.RGB_from_LMS @ sim_matrix @ self.LMS_from_RGB

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        linRGB = LinRGB.from_sRGB(image)
        inv_linRGB = torch.einsum("ij,bjhw->bihw", self.matrix, linRGB)
        return LinRGB.to_sRGB(inv_linRGB)
