from __future__ import annotations
from typing import Literal, Annotated, Callable
from pydantic import BaseModel, ConfigDict, Field
from pathlib import Path
import json
import torch
from torch import Tensor

from .daltonization.simulation import VienotSimulation, FarupSimulation
from .daltonization.loss import (
    VienotSignGuide,
    FarupSignGuide,
    FullProblemOptimization,
    ReducedProblemOptimization,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FullProblemOpt(StrictModel):
    name: Literal["full_problem_optimization"]
    pixel_bias: int = Field(1)
    eps: float = Field(15e-3)

    def create(self, x: VienotSimulation | FarupSimulation):
        return FullProblemOptimization(x, self.pixel_bias, self.eps)


class ReducedProblemOpt(StrictModel):
    name: Literal["reduced_problem_optimization"]
    pixel_bias: int = Field(1)
    eps: float = Field(15e-3)
    avg_ma: float = Field(0.8)

    sign_guide: Literal["l", "b", "lb", "lbb", "s_l"]

    def create(self, x: VienotSimulation | FarupSimulation):
        if isinstance(x, VienotSimulation):
            if self.sign_guide in ["b", "l", "lb", "s_l"]:
                guid_func = VienotSignGuide(x.cvd_type, self.sign_guide)
            else:
                raise KeyError
        elif isinstance(x, FarupSimulation):
            if self.sign_guide in ["b", "lb", "lbb"]:
                guid_func = FarupSignGuide(x.cvd_type, self.sign_guide)
                return ReducedProblemOptimization(
                    x,
                    guid_func,
                )
            else:
                raise KeyError
        return ReducedProblemOptimization(
            x, guid_func, self.avg_ma, self.eps, self.pixel_bias
        )


class AdamConfig(StrictModel):
    name: Literal["Adam"]
    learning_rate: float = 1e-5

    def load(
        self,
    ) -> Callable[[Tensor], torch.optim.Optimizer]:
        def optimizer(weight_map: Tensor):
            return torch.optim.Adam([weight_map], lr=self.learning_rate)

        return optimizer


class VienotConfig(StrictModel):
    name: Literal["Vienot"]
    cvd_type: Literal["protan", "deutan"]

    def create(self, device) -> VienotSimulation:

        return VienotSimulation(self.cvd_type, device)


class FarupConfig(StrictModel):
    name: Literal["Farup"]
    cvd_type: Literal["rg", "by"]

    def create(self, device) -> FarupSimulation:

        return FarupSimulation(self.cvd_type, device)


class Config(StrictModel):
    simulation: Annotated[
        VienotConfig | FarupConfig, Field(..., discriminator="name")
    ]
    loss: Annotated[
        FullProblemOpt | ReducedProblemOpt, Field(..., discriminator="name")
    ]
    optimizer: Annotated[AdamConfig, Field(..., discriminator="name")] = (
        AdamConfig(name="Adam")
    )
    dataset_path: Path
    save_dir: Path
    batch_size: int = 11
    epochs: int = 10000
    gap: float = 1e-8
    cuda: int = 0

    @classmethod
    def read_config(cls, config_path: Path):
        return cls(**json.load(config_path.open()))
