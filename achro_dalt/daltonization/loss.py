from __future__ import annotations
from typing import Literal, List, Callable
import torch
from .simulation import VienotSimulation, FarupSimulation


def _grad(
    image: torch.Tensor, px_bias: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image - torch.roll(image, shifts=px_bias, dims=-1)
    dy = image - torch.roll(image, shifts=px_bias, dims=-2)
    return (dx, dy)


def _mean(
    image: torch.Tensor, px_bias: int
) -> tuple[torch.Tensor, torch.Tensor]:
    mean_x = (image + torch.roll(image, shifts=px_bias, dims=-1)) / 2.0
    mean_y = (image + torch.roll(image, shifts=px_bias, dims=-2)) / 2.0
    return (mean_x, mean_y)


class FullProblemOptimization:
    def __init__(
        self,
        sim_func: VienotSimulation | FarupSimulation,
        px_bias: int = 1,
        eps: float = 0.015,
    ) -> None:
        self.sim_func = sim_func
        self.px_bias = px_bias
        self.eps = eps

    def __call__(
        self,
        image: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        grad_x, grad_y = _grad(image, self.px_bias)
        inNorm2_x = torch.sum(grad_x**2, dim=1)
        inNorm2_y = torch.sum(grad_y**2, dim=1)

        def loss(x: torch.Tensor) -> torch.Tensor:
            simulated = self.sim_func(
                image.permute(0, 2, 3, 1) * x.unsqueeze(-1)
            )
            dx_sim, dy_sim = _grad(simulated, self.px_bias)
            simNorm2_x = torch.sum(dx_sim**2, dim=3)
            simNorm2_y = torch.sum(dy_sim**2, dim=3)

            matErr = (
                (simNorm2_x + 1e-32) ** (1 / 2)
                - (inNorm2_x + 1e-32) ** (1 / 2)
            ) ** 2 / (inNorm2_x + self.eps**2) + (
                (simNorm2_y + 1e-32) ** (1 / 2)
                - (inNorm2_y + 1e-32) ** (1 / 2)
            ) ** 2 / (
                inNorm2_y + self.eps**2
            )

            return matErr.mean(dim=(1, 2)).sum()

        return loss


class ReducedProblemOptimization:
    def __init__(
        self,
        sim_func: VienotSimulation | FarupSimulation,
        sign_func: VienotSignGuide | FarupSignGuide,
        average_value: float = 0.8,
        eps: float = 0.015,
        px_bias: int = 1,
        thresh: float = 0.0,
        device: int = 0,
    ) -> None:
        self.sim_func = sim_func
        self.sign_func = sign_func
        self.average_value = average_value
        self.eps = eps
        self.px_bias = px_bias
        self.thresh = thresh
        self.device = device

    def __call__(
        self,
        image: torch.Tensor,
    ) -> Callable[[torch.Tensor], torch.Tensor]:

        sign_guide = self.sign_func(image, self.device)
        sign_x, sign_y = _grad(sign_guide, self.px_bias)
        # perm_batch = image.permute(0, 2, 3, 1)
        grad_x, grad_y = _grad(image, self.px_bias)
        mean_x, mean_y = _mean(image, self.px_bias)

        dw_x = self._get_dw(
            grad_x,
            mean_x,
            sign_x,
            self.sim_func,
            self.average_value,
            self.thresh,
        )
        dw_y = self._get_dw(
            grad_y,
            mean_y,
            sign_y,
            self.sim_func,
            self.average_value,
            self.thresh,
        )

        def loss(x: torch.Tensor) -> torch.Tensor:
            dx, dy = _grad(x, self.px_bias)
            matErr = (dx - dw_x) ** 2 / (dw_x**2 + self.eps**2) + (
                dy - dw_y
            ) ** 2 / (dw_y**2 + self.eps**2)
            return matErr.mean(dim=(-1, -2)).sum()

        return loss

    def _get_dw(
        self,
        grad: torch.Tensor,
        mean: torch.Tensor,
        sign: torch.Tensor,
        sim_func: VienotSimulation | FarupSimulation,
        average_value: float,
        thresh: float,
    ):
        a = sim_func(mean)
        b = average_value * sim_func(grad)
        c = torch.sum(grad**2, dim=1)

        D_pt1 = 2 * torch.sum(a * b, dim=1)
        D_pt2 = 4 * torch.sum(a**2, dim=1) * (torch.sum(b**2, dim=1) - c)

        D = D_pt1**2 - D_pt2

        print(f"D < 0: {(D < 0).sum()}")
        print(f"DMax: {(D[D <= 0]).min()}")
        print(D[D < 0])

        D[D < 0] = 0

        dw1 = (-D_pt1 + torch.sqrt(D)) / (2 * torch.sum(a**2, dim=1))
        dw2 = (-D_pt1 - torch.sqrt(D)) / (2 * torch.sum(a**2, dim=1))

        mask_lesser_dw = sign < -thresh
        mask_larger_dw = sign > thresh

        dw_larger = torch.where(dw1 >= dw2, dw1, dw2)
        dw_lesser = torch.where(dw1 <= dw2, dw1, dw2)
        out = torch.zeros_like(dw1)
        out[mask_larger_dw] = dw_larger[mask_larger_dw]
        out[mask_lesser_dw] = dw_lesser[mask_lesser_dw]
        return out


class FarupSignGuide:
    vec4blnd: List[float]

    def __init__(
        self,
        cvd_type: Literal["rg", "by"],
        sign_guide: Literal["lb", "lbb", "b"],
    ) -> None:
        if cvd_type == "rg":
            if sign_guide == "lb":
                self.vec4blnd = [2.0, 0.0, 1.0]
            elif sign_guide == "lbb":
                self.vec4blnd = [1.5, -0.5, 0.5]
            elif sign_guide == "b":
                self.vec4blnd = [1.0, -1.0, 0.0]
            else:
                raise KeyError(cvd_type, "hasn't option:", sign_guide)
        elif cvd_type == "by":
            if sign_guide == "b":
                self.vec4blnd = [-1.0, -1.0, 1.0]
            else:
                raise KeyError(cvd_type, "hasn't option:", sign_guide)
        else:
            raise KeyError("unknown cvd type:", cvd_type)

        self.cvd_type = cvd_type
        self.sign_guide = sign_guide

    def __call__(self, image: torch.Tensor, device) -> torch.Tensor:
        blind_3d = image.permute(0, 2, 3, 1)
        sg = torch.mean(blind_3d * self.vec4blnd, dim=0)
        if self.cvd_type == "by":
            return sg + 2.0 / 3.0
        return sg


class VienotBlind(VienotSimulation):
    protan_matrix = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    deutan_matrix = torch.tensor(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )


class VienotSignGuide:
    cvd_type: Literal["protan", "deutan"]
    sign_guide: Literal["l", "b", "lb", "s_l"]

    def __init__(
        self,
        cvd_type: Literal["protan", "deutan"],
        sign_guide: Literal["l", "b", "lb", "s_l"],
    ) -> None:
        assert cvd_type in ["protan", "deutan"]
        assert sign_guide in ["l", "b", "lb", "s_l"]
        self.cvd_type = cvd_type
        self.sign_guide = sign_guide

    def __call__(self, image: torch.Tensor, device) -> torch.Tensor:
        blind_3d = image.permute(0, 2, 3, 1)
        if self.sign_guide == "l":
            return blind_3d.mean(dim=3)
        elif self.sign_guide == "b":
            blind_sim = VienotBlind(self.cvd_type, device)(blind_3d)
            return blind_sim.mean(dim=3)
        elif self.sign_guide == "lb":
            blind_sim = VienotBlind(self.cvd_type, device)(blind_3d)
            return (blind_sim.mean(dim=3) + blind_3d.mean(dim=3)) / 2.0
        elif self.sign_guide == "s_l":
            blind_sim = VienotSimulation(self.cvd_type, device)(blind_3d)
            return blind_sim.mean(dim=3)
