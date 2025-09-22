from __future__ import annotations

import torch
from typing import Callable
from pathlib import Path
from tqdm import tqdm

from .dataset import create_dataloader, save_results


def local_change_range(image: torch.Tensor, percentile: float):
    assert 0 < percentile < 1.0
    max_channel = image.amax(dim=1)
    divisor = torch.quantile(
        max_channel.view(max_channel.shape[0], -1), percentile, dim=1
    )
    return (image / divisor[:, None, None, None]).clip(0, 1)


def apply_watermark(
    image: torch.Tensor, watermark: torch.Tensor
) -> torch.Tensor:
    watermark -= torch.amin(watermark, dim=(-1, -2))[:, None, None]
    return local_change_range(image * watermark[:, None, :, :], 0.98)


def optimize(
    sim_func: Callable[[torch.Tensor], torch.Tensor],
    create_optimizer: Callable[[torch.Tensor], torch.optim.Optimizer],
    create_loss: Callable[
        [torch.Tensor], Callable[[torch.Tensor], torch.Tensor]
    ],
    epochs: int,
    gap: float,
    dataset_path: Path,
    batch_size: int,
    save_dir: Path,
    device,
):
    dataloader = create_dataloader(dataset_path, batch_size)
    output_paths = list()
    for inputs, sizes, names in dataloader:
        b, c, h, w = inputs.shape
        x = torch.ones((b, h, w), requires_grad=True, device="cpu")

        optimizer = create_optimizer(x)
        loss_func = create_loss(inputs)

        init_loss = loss_func(x)
        print(f"Initial Loss: {init_loss.item()}")

        prev_loss = torch.tensor(float("inf"))

        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            loss = loss_func(x)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % (epochs // 10) == 0:
                print(f"Mean Loss: {loss.item()}")

            if torch.abs(loss - prev_loss) < gap:
                print(f"Mean Loss: {loss.item()}")
                break

        dalt = apply_watermark(inputs, x.detach())
        output_paths.extend(save_results(dalt, sizes, names, save_dir, "dalt"))
        save_results(sim_func(inputs), sizes, names, save_dir, "orig_sim")
        save_results(sim_func(dalt), sizes, names, save_dir, "dalt_sim")
    return output_paths
